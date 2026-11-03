import sys, collections, re
c=collections.Counter(); n=0
for l in open(sys.argv[1]):
    l=l.strip()
    if not l: continue
    n+=1
    m=[x for x in l.split(";") if "@" in x]
    if m: c[m[-1].split("@")[1]]+=1
    else: c["(no menos frame)"]+=1
print("samples",n)
for k,v in c.most_common(int(sys.argv[2]) if len(sys.argv)>2 else 15): print(f"{100*v/n:5.1f}%  {k}")
