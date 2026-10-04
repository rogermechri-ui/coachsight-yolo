import sys,time,pickle; E=sys.argv[1]; sys.path.insert(0,E)
import cv2,numpy as np
from cam import *
C=np.array([float(x) for x in sys.argv[2].split(',')])
ids=[int(x) for x in sys.argv[3].split(',')]
out={}
for k in ids:
    fr=Frame(cv2.imread('%s/f%02d.jpg'%(E,k))); t=time.time()
    th,ph,f,v=solve2(fr,C)
    H=homog(C,th,ph,f,fr.cx,fr.cy); c=cost_bi(fr,H)
    out[k]=(th,ph,f,c)
    print(k,'pan %.1f tilt %.1f f %d'%(np.degrees(th),np.degrees(ph),f),'cost',[round(x,2) for x in c],'%.0fs'%(time.time()-t),flush=True)
    cv2.imwrite('%s/r%02d.jpg'%(E,k),pf.draw_overlay(fr.img,H,length=L,width=W))
pickle.dump(out,open(E+'/run_%s.pkl'%sys.argv[4],'wb'))
