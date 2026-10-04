import sys,os,pickle,time; E=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,E)
import cv2,numpy as np
from cam import *
C=eval(sys.argv[1]); ks=[int(x) for x in sys.argv[2].split(',')]; tag=sys.argv[3]
out={}
for k in ks:
    im=cv2.imread('%s/%s%04d.jpg'%(E,os.environ.get('PFX','v'),k)); fr=Frame(im); t=time.time()
    th,ph,f,v=solve2(fr,C); out[k]=(th,ph,f,v)
    print(k,'pan %.1f tilt %.1f f %d v %.2f %.0fs'%(np.degrees(th),np.degrees(ph),f,v,time.time()-t),flush=True)
    cv2.imwrite('%s/o%04d_%s.jpg'%(E,k,tag),draw_lines(im,homog(C,th,ph,f,fr.cx,fr.cy)))
pickle.dump(out,open('%s/near_%s.pkl'%(E,tag),'wb'))
