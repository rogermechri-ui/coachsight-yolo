import sys,time,pickle; sys.path.insert(0,sys.argv[1])
import cv2,numpy as np
from cam import *
from scipy.optimize import minimize
E=sys.argv[1]
ids=[1,3,5,7,12,9]
frs={k:Frame(cv2.imread('%s/f%02d.jpg'%(E,k))) for k in ids}
C=np.array([52.5,68+5,7.0])
sol={}
for k in ids:
    sol[k]=solve(frs[k],C)[:3]; print(k,np.degrees(sol[k][:2]).round(1),round(sol[k][2]),flush=True)
for it in range(4):
    def objC(c):
        return sum(sum(cost_bi(frs[k],homog(c,*sol[k][:2],sol[k][2],frs[k].cx,frs[k].cy))) for k in ids)
    r=minimize(objC,C,method='Powell',options={'xtol':0.05,'ftol':1e-3,'maxfev':300})
    C=r.x; print('iter',it,'C',C.round(2),'cost',round(r.fun/len(ids),2),flush=True)
    for k in ids:
        th,ph,f,v=refine(frs[k],C,*sol[k]); sol[k]=(th,ph,f)
pickle.dump((C,sol),open(E+'/calib.pkl','wb'))
for k in ids:
    H=homog(C,*sol[k][:2],sol[k][2],frs[k].cx,frs[k].cy)
    print(k,[round(x,2) for x in cost_bi(frs[k],H)])
    cv2.imwrite('%s/c%02d.jpg'%(E,k),pf.draw_overlay(frs[k].img,H,length=L,width=W))
