
 python test_latency_remote.py --reps 3     每个动作 3 次

七个动
import argparse
import csv
import json
import statistics
import time
import urllib.request

BASE = "http://8.138.114.112:8000"
KEY = "e373e7b39ee0174c1fefeb5cf2a66bc2"

# 动作, 参数, 发完等多久再发下一条（秒）
ACTIONS = [
    ("walk_forward",    {"vx": 0.3},  3),
    ("walk_backward",   {"vx": -0.4}, 3),
    ("turn_left",       {"wz": 1.0},  3),
    ("turn_right",      {"wz": -1.0}, 3),
    ("stop",            {},           1),
    ("toggle_sitstand", {},           6),
    ("dance",           {},           5),
]

ENDED = ("finished", "completed", "failed", "error", "rejected", "timeout")


def call(base, key, path, body=None):
    """请求后端。返回 (解析出的字典, 耗时毫秒)，出错时字典是 None。"""
    headers = {"X-API-Key": key}
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(base + path, data=data, headers=headers)
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=6) as r:
            return json.loads(r.read() or b"{}"), (time.time() - t0) * 1000
    except Exception:
        return None, (time.time() - t0) * 1000


def server_ms(obj):
    """后端自己记的 created_at 到 completed_at，没有就返回 None。"""
    a, b = (obj or {}).get("created_at"), (obj or {}).get("completed_at")
    if not a or not b:
        return None
    try:
        from datetime import datetime
        return (datetime.fromisoformat(b) - datetime.fromisoformat(a)).total_seconds() * 1000
    except Exception:
        return None


def run_once(base, key, op, params, deadline):
    """发一条命令，等它跑完。返回 (api_ms, e2e_ms, server_ms, status)。"""
    res, api_ms = call(base, key, "/api/v1/commands", {"op": op, "params": params})
    if not res or not res.get("command_id"):
        return None, None, None, "send_error"

    cid = res["command_id"]
    t0 = time.time()
    status, obj = "", {}
    while time.time() - t0 < deadline:
        obj, _ = call(base, key, "/api/v1/commands/" + cid)
        status = (obj or {}).get("status", "")
        if status in ENDED:
            break
        time.sleep(0.05)

    e2e_ms = api_ms + (time.time() - t0) * 1000
    return api_ms, e2e_ms, server_ms(obj), (status or "timeout")


def check(args):
    obj, ms = call(args.base, args.key, "/api/v1/status")
    if obj is None:
        print("连不上后端，或者 API Key 不对")
        return
    print("后端响应 %.0f ms，当前状态 %s" % (ms, obj.get("status")))
    if obj.get("status") == "online":
        print("仿真端在线，可以开跑")
    else:
        print("仿真端离线，命令发出去没人执行，先确认仿真那台电脑在跑")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10, help="每个动作跑几次")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--key", default=KEY)
    ap.add_argument("--deadline", type=float, default=10, help="一条命令最多等几秒")
    ap.add_argument("--settle", type=float, default=2, help="两次测试之间再停几秒")
    ap.add_argument("--out", default="result_latency.csv")
    ap.add_argument("--check", action="store_true", help="只检查连通性")
    args = ap.parse_args()

    if args.check:
        check(args)
        return

    plan = [ACTIONS[i % len(ACTIONS)] for i in range(len(ACTIONS) * args.reps)]
    print("共 %d 次，开始 %s" % (len(plan), time.strftime("%Y-%m-%d %H:%M:%S")))

    f = open(args.out, "w", newline="", encoding="utf-8-sig")
    w = csv.writer(f)
    w.writerow(["trial", "action", "step", "time", "api_ms", "e2e_ms", "server_ms", "status"])

    stat = {}          # 动作名 -> [成功数, 总数, e2e 列表]
    try:
        for i, (op, params, wait_after) in enumerate(plan, 1):
            steps = ("蹲下", "起立") if op == "toggle_sitstand" else ("",)
            for step in steps:
                api_ms, e2e_ms, srv, status = run_once(
                    args.base, args.key, op, params, args.deadline)
                ok = status in ENDED and status not in ("failed", "error", "timeout", "rejected")

                name = op + ("(" + step + ")" if step else "")
                rec = stat.setdefault(name, [0, 0, []])
                rec[1] += 1
                if ok:
                    rec[0] += 1
                    rec[2].append(e2e_ms)

                print("  [%2d/%d] %-20s api %6s  e2e %7s  %s"
                      % (i, len(plan), name,
                         "-" if api_ms is None else "%.1f" % api_ms,
                         "-" if e2e_ms is None else "%.1f" % e2e_ms,
                         status))
                w.writerow([i, op, step, time.strftime("%H:%M:%S"),
                            "" if api_ms is None else "%.1f" % api_ms,
                            "" if e2e_ms is None else "%.1f" % e2e_ms,
                            "" if srv is None else "%.1f" % srv,
                            status])
                f.flush()
                time.sleep(wait_after)
            time.sleep(args.settle)
    except KeyboardInterrupt:
        print("手动停了")
    finally:
        f.close()

    print()
    print("%-24s %-10s %-12s %-12s" % ("动作", "完成/次数", "e2e中位数", "e2e最大"))
    for op, _, _ in ACTIONS:
        for step in (("蹲下", "起立") if op == "toggle_sitstand" else ("",)):
            name = op + ("(" + step + ")" if step else "")
            good, total, vals = stat.get(name, [0, 0, []])
            print("%-24s %-10s %-12s %-12s"
                  % (name, "%d/%d" % (good, total),
                     "-" if not vals else "%.1f ms" % statistics.median(vals),
                     "-" if not vals else "%.1f ms" % max(vals)))
    print("\n结果文件 %s" % args.out)


main()
