import sys,pickle,os; E=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,E)
import cv2,numpy as np
from cam import *
clip,tag,C=sys.argv[1],sys.argv[2],eval(sys.argv[3]); idx=[int(x) for x in sys.argv[4].split(',')]
res=pickle.load(open('%s/track_%s.pkl'%(E,tag),'rb'))
sc=[res[i][1] for i in sorted(res)]
print('score by 50:',' '.join('%d:%.2f'%(i,np.mean(sc[i:i+50])) for i in range(0,len(sc),50)))
cap=cv2.VideoCapture(clip); tiles=[]
for i in idx:
    cap.set(cv2.CAP_PROP_POS_FRAMES,i); ok,im=cap.read(); p=res[i][0]
    fr_cx,fr_cy=(163+1117)/2,im.shape[0]/2
    v=pf.draw_overlay(im,homog(C,p[0],p[1],p[2],fr_cx,fr_cy),length=L,width=W)[:,163:1117]
    cv2.putText(v,'%d  s=%.2f'%(i,res[i][1]),(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,0,255),2); tiles.append(v)
while len(tiles)%2: tiles.append(np.zeros_like(tiles[0]))
g=np.vstack([np.hstack(tiles[i:i+2]) for i in range(0,len(tiles),2)])
cv2.imwrite('%s/show_%s.jpg'%(E,tag),cv2.resize(g,None,fx=0.6,fy=0.6))
