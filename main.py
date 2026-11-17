#!/usr/bin/env python3
import os
import glob
import json
import math
from datetime import datetime, timedelta
from statistics import stdev
from decimal import Decimal, ROUND_HALF_UP

Z_VALS = [
    -1.6448536269514722,
    -0.6744897501960817,
    0.0,
    0.6744897501960817,
    1.6448536269514722
]
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

def round_half_up(val):
    d = Decimal(str(val))
    rounded = d.quantize(Decimal('0.01'), rounding=ROUND_HALF_UP)
    if rounded == rounded.to_integral_value():
        return int(rounded)
    else:
        return float(rounded)

def process_case(case_data):
    """
    Forecasts hourly Twitch CCU quantiles for GTA VI launch using a kernel
    local-linear log-ratio regression model trained on comparable game launches,
    calibrated with leave-one-out residual standard deviation.
    """
    # 1. Extract p1 (hour 1 viewers)
    p1 = 1
    known_list = case_data.get("known", [])
    h1_row = next((r for r in known_list if r.get("hour") == 1), None)
    if h1_row and "viewers" in h1_row:
        p1 = h1_row["viewers"]
    else:
        history_hourly = case_data.get("history_hourly", [])
        if history_hourly:
            p1 = history_hourly[-1].get("viewers", 1)
        else:
            history = case_data.get("history", [])
            if history:
                p1 = history[-1].get("viewers", 1)
    if p1 is None or p1 < 1:
        p1 = 1

    log_p1 = math.log(p1)

    # 2. Filter comparables
    comparables = case_data.get("comparables", [])
    usable = []
    for c in comparables:
        day_start_str = c.get("day_start")
        if not day_start_str:
            continue
        try:
            ds = datetime.fromisoformat(day_start_str.replace("Z", "+00:00"))
        except Exception:
            continue
        
        hourly = c.get("hourly", [])
        h_map = {}
        for item in hourly:
            f_str = item.get("from")
            v = item.get("viewers")
            if f_str and v is not None:
                try:
                    dt = datetime.fromisoformat(f_str.replace("Z", "+00:00"))
                    h_map[dt] = v
                except Exception:
                    pass

        ok = True
        v1 = None
        h_vals = {}
        for h in range(1, 25):
            t_dt = ds + timedelta(hours=h-1)
            if t_dt not in h_map:
                ok = False
                break
            v = h_map[t_dt]
            if h == 1:
                v1 = v
            else:
                h_vals[h] = v
        
        if ok and v1 is not None and v1 > 0:
            usable.append({
                "v1": v1,
                "log_v1": math.log(v1),
                "h_vals": h_vals
            })

    hours_output = []

    # Fallback if fewer than 3 usable comparables exist at all
    if len(usable) < 3:
        for h in range(2, 25):
            median_h = float(p1)
            sigma_h = 0.15 * math.sqrt(h - 1)
            q_list = []
            prev_q = 0.0
            for z in Z_VALS:
                q = median_h * math.exp(z * sigma_h)
                q = max(1.0, min(1e9, q))
                if q < prev_q:
                    q = prev_q
                prev_q = q
                q_list.append(round_half_up(q))
            hours_output.append({"hour": h, "q": q_list})
        return {"hours": hours_output}

    # For each forecast hour h = 2..24
    for h in range(2, 25):
        # Training comparables with data for hour h
        train_set = [u for u in usable if h in u["h_vals"] and u["v1"] > 0]
        
        if len(train_set) < 3:
            median_h = float(p1)
            sigma_h = 0.30
        else:
            xs = [u["log_v1"] for u in train_set]
            ys = [math.log(u["h_vals"][h] / u["v1"]) for u in train_set]
            
            weights = [math.exp(-0.5 * ((x - log_p1) / 1.5) ** 2) for x in xs]
            W = sum(weights)
            if W == 0:
                b = 0
                a = sum(ys) / len(ys) if ys else 0
            else:
                mean_x = sum(w * x for w, x in zip(weights, xs)) / W
                mean_y = sum(w * y for w, y in zip(weights, ys)) / W
                sxx = sum(w * (x - mean_x)**2 for w, x in zip(weights, xs))
                sxy = sum(w * (x - mean_x) * (y - mean_y) for w, x, y in zip(weights, xs, ys))
                if sxx < 1e-12:
                    b = 0
                    a = mean_y
                else:
                    b = sxy / sxx
                    a = mean_y - b * mean_x

            min_x = min(xs)
            max_x = max(xs)
            xq = min(max(log_p1, min_x), max_x)
            median_h = p1 * math.exp(a + b * xq)

            # LOO residual std dev
            residuals = []
            for i, train_item in enumerate(train_set):
                xi = train_item["log_v1"]
                yi = math.log(train_item["h_vals"][h] / train_item["v1"])
                other_items = [u for k, u in enumerate(train_set) if k != i]
                other_xs = [u["log_v1"] for u in other_items]
                other_ys = [math.log(u["h_vals"][h] / u["v1"]) for u in other_items]
                
                w_loo = [math.exp(-0.5 * ((x - xi) / 1.5) ** 2) for x in other_xs]
                W_loo = sum(w_loo)
                if W_loo == 0:
                    b_loo = 0
                    a_loo = sum(other_ys) / len(other_ys) if other_ys else 0
                else:
                    mx_loo = sum(w * x for w, x in zip(w_loo, other_xs)) / W_loo
                    my_loo = sum(w * y for w, y in zip(w_loo, other_ys)) / W_loo
                    sxx_loo = sum(w * (x - mx_loo)**2 for w, x in zip(w_loo, other_xs))
                    sxy_loo = sum(w * (x - mx_loo) * (y - my_loo) for w, x, y in zip(w_loo, other_xs, other_ys))
                    if sxx_loo < 1e-12:
                        b_loo = 0
                        a_loo = my_loo
                    else:
                        b_loo = sxy_loo / sxx_loo
                        a_loo = my_loo - b_loo * mx_loo
                pred_y_i = a_loo + b_loo * xi
                residuals.append(yi - pred_y_i)

            if len(residuals) < 3:
                sd_h = 0.30
            else:
                sd_h = stdev(residuals)
            sigma_h = max(0.8 * sd_h, 0.12)

        q_list = []
        prev_q = 0.0
        for z in Z_VALS:
            q = median_h * math.exp(z * sigma_h)
            q = max(1.0, min(1e9, q))
            if q < prev_q:
                q = prev_q
            prev_q = q
            q_list.append(round_half_up(q))
        
        hours_output.append({"hour": h, "q": q_list})

    return {"hours": hours_output}

def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input/")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output/")
    os.makedirs(output_dir, exist_ok=True)

    in_pattern = os.path.join(input_dir, "*.in")
    in_files = sorted(glob.glob(in_pattern))

    for in_path in in_files:
        stem = os.path.splitext(os.path.basename(in_path))[0]
        out_path = os.path.join(output_dir, f"{stem}.out")

        with open(in_path, "r", encoding="utf-8") as f:
            case_data = json.load(f)

        result = process_case(case_data)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)

if __name__ == "__main__":
    main()
