import sys, collections, re
stacks=[l.strip().split(";") for l in open(sys.argv[1]) if l.strip()]
n=len(stacks)
def short(f):
    return f.replace("(anonymous namespace)::","anon::")
incl=collections.Counter(); selfc=collections.Counter()
for s in stacks:
    seen=set()
    for f in s:
        f=short(f)
        if f not in seen: incl[f]+=1; seen.add(f)
    # self: innermost menos-ish frame
    inner=[short(f) for f in s if "menos" in f]
    selfc[short(s[-1])]+=1
print(f"samples={n}")
print("-- inclusive (menos frames) --")
for f,c in incl.most_common(60):
    if "menos" in f or "lapack" in f.lower() or "zgesvd" in f: print(f"{100*c/n:5.1f}%  {f}")
print("-- self (innermost frame) --")
for f,c in selfc.most_common(12): print(f"{100*c/n:5.1f}%  {f}")
