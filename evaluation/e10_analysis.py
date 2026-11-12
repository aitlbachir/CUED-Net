import sys, json
sys.path.insert(0, sys.argv[1])  # repository evaluation folder
import numpy as np
import p2_analysis as P
runs, out, B = sys.argv[2], sys.argv[3], int(sys.argv[4])
names = ["cued_net", "cued_resnet50", "cued_effnet_b0", "cued_no_consistency", "cued_vdl"]
tables = {m: (P.add_scores(P.read_split(runs, m, "inner_val")), P.add_scores(P.read_split(runs, m, "outer_val"))) for m in names}
models = {m: P.Prepared(iv, ov, unc="u_total") for m, (iv, ov) in tables.items()}
ref = models["cued_net"]
for k, v in models.items():
    assert v.keys == ref.keys, k
    print(k, "seeds", v.seeds, "cases", len(v.keys))
point, dist = P.bootstrap_all(models, ref.group, B)
summ = P.summarise(point, dist)
cmp_cls = P.compare(point, dist, "cued_net", names[1:])
cmp_rank = P.compare(point, dist, "cued_net", names[1:], P.RANKING)
vt = {m: P.view_term(iv, ov, B) for m, (iv, ov) in tables.items()}
res = dict(B=B, n_cases=len(ref.keys), summary={m: {x: summ[m][x] for x in ("auc", "f1", "aurc", "eaurc", "ece", "ece_cal", "sensitivity", "specificity")} for m in names},
           comparison={**cmp_cls, **cmp_rank}, view_term=vt,
           temperature={m: float(np.mean(v.temperatures)) for m, v in models.items()})
json.dump(res, open(out, "w"), indent=1)
for m in names:
    s = summ[m]; print(m, {x: f"{s[x]['value']:.3f} [{s[x]['ci95'][0]:.3f}, {s[x]['ci95'][1]:.3f}]" for x in ("auc", "f1", "aurc", "ece_cal")})
