#1. 사진 영역 cut 이미지 생성기 만들기(파일로드, cut, 저장)
#2. Ex2동작으로 배경제거
#3. 대상 검출기 만들기(검출대상 로드, 검출정보 로드, 찾기이미지 생성, 저장)
#번외 영상으로 접근
#pyQt를 이용하여 프로그램을 완성하시오
from PyQt5.QtWidgets import *
import sys
import cv2 
import numpy as np

class c_n (QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("그림_자르기")
        self.setGeometry(200,200,420,100)
        f_l_b = QPushButton("파일 로드",self)
        c_b = QPushButton("자르기",self)
        f_s_b = QPushButton("파일저장",self)
        e_b = QPushButton("종료",self)
        self.label = QLabel("프로그램이 켜집니다.",self)

        f_l_b.setGeometry(10,10,100,30)
        c_b.setGeometry(110,10,100,30)
        f_s_b.setGeometry(210,10,100,30)
        e_b.setGeometry(310,10,100,30)
        self.label.setGeometry(10,50,200,30)

        f_l_b.clicked.connect(self.f_l_b_f)
        c_b.clicked.connect(self.c_b_f)
        f_s_b.clicked.connect(self.f_s_b_f)
        e_b.clicked.connect(self.e_b_f)

        self.SX=0
        self.SY=0
        self.EX=0
        self.EY=0

    def f_l_b_f(self):
        #파일 로드
        l_fname=QFileDialog.getOpenFileName(self,"파일 로드",'./')
        self.img=cv2.imread(l_fname[0])
        if self.img is None:
            self.label.setText("파일 로드 실패")
            return
        self.label.setText("파일 로드 성공")
        cv2.imshow("show_img",self.img)
        
        
         
    def draw(self,event,x,y,f,p):
        if event==cv2.EVENT_LBUTTONDOWN:
            self.SX,self.SY=x,y
        elif event==cv2.EVENT_LBUTTONUP:
            self.EX,self.EY=x,y
            self.cut_img=self.img[self.SY:self.EY,self.SX:self.EX]
            
        cv2.imshow("show_img",self.img)
        cv2.imshow("cut_img",self.cut_img)

    def c_b_f(self):
        #오리기
        cv2.setMouseCallback('show_img',self.draw)
        self.cut_img=np.copy(self.img)
        self.label.setText("절단을 확정하려면 c를 누르시오")
        while True:
            if cv2.waitKey(1)==ord('c'):
                cv2.destroyAllWindows()
                break

    def f_s_b_f(self):
        #파일 저장
        s_fname=QFileDialog.getSaveFileName(self,"파일 저장",'./')
        cv2.imwrite(s_fname[0],self.cut_img)
        self.label.setText("저장 완료")

    def e_b_f(self):
        #종료
        
        cv2.destroyAllWindows()
        self.close()
        m_win

class Cut_bk_img (QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("그림_제거")
        self.setGeometry(200,200,620,100)

        f_l_b = QPushButton("파일 로드",self)
        b_i_b = QPushButton("붓그리기",self)
        p_b = QPushButton("+",self)
        m_b = QPushButton("-",self)
        c_b = QPushButton("잘라내기",self)
        f_s_b = QPushButton("파일저장",self)
        e_b = QPushButton("종료",self)
        self.label = QLabel("프로그램이 켜집니다.",self)
        self.p_label = QLabel(" ",self)

        f_l_b.setGeometry(10,10,100,30)
        b_i_b.setGeometry(110,10,100,30)
        p_b.setGeometry(210,10,50,30)
        m_b.setGeometry(260,10,50,30)
        c_b.setGeometry(310,10,100,30)
        f_s_b.setGeometry(410,10,100,30)
        e_b.setGeometry(510,10,100,30)
        self.label.setGeometry(10,50,200,30)
        self.p_label.setGeometry(210,50,200,30)

        f_l_b.clicked.connect(self.f_l_b_f)
        b_i_b.clicked.connect(self.b_i_b_f)
        p_b.clicked.connect(self.p_f)
        m_b.clicked.connect(self.m_f)
        c_b.clicked.connect(self.c_b_f)
        f_s_b.clicked.connect(self.f_s_b_f)
        e_b.clicked.connect(self.e_b_f)

        self.L_C,self.R_C=(0,0,255),(255,0,0)
        self.P_SIZE=5

    def p_f(self):
        self.P_SIZE=min(30,self.P_SIZE+1)
        self.p_label.setText(f"{self.P_SIZE}")

    def m_f(self):
        self.P_SIZE=max(1,self.P_SIZE-1)
        self.p_label.setText(f"{self.P_SIZE}")

    def f_l_b_f(self):
        #파일 로드
        l_fname=QFileDialog.getOpenFileName(self,"파일 로드",'./')
        self.img=cv2.imread(l_fname[0])
        if self.img is None:
            #self.label.setText("파일이 없습니다.")
            sys.exit("파일이 없습니다.")
        self.label.setText("파일 로드 성공")
        self.show_img=np.copy(self.img)
        cv2.imshow("show_img",self.show_img)

        self.mask=np.zeros((self.img.shape[0],self.img.shape[1]),np.uint8)
        self.mask[:,:]=cv2.GC_PR_BGD
    def dw_b(self,event,x,y,f,p):
        if event == cv2.EVENT_LBUTTONDOWN:
            cv2.circle(self.show_img,(x,y),self.P_SIZE,self.L_C,-1)
            cv2.circle(self.mask,(x,y),self.P_SIZE,cv2.GC_FGD,-1)
        elif event == cv2.EVENT_RBUTTONDOWN:
            cv2.circle(self.show_img,(x,y),self.P_SIZE,self.R_C,-1)
            cv2.circle(self.mask,(x,y),self.P_SIZE,cv2.GC_BGD,-1)
        elif event == cv2.EVENT_MOUSEMOVE and f==cv2.EVENT_FLAG_LBUTTON:
            cv2.circle(self.show_img,(x,y),self.P_SIZE,self.L_C,-1)
            cv2.circle(self.mask,(x,y),self.P_SIZE,cv2.GC_FGD,-1)
        elif event == cv2.EVENT_MOUSEMOVE and f==cv2.EVENT_FLAG_RBUTTON:
            cv2.circle(self.show_img,(x,y),self.P_SIZE,self.R_C,-1)
            cv2.circle(self.mask,(x,y),self.P_SIZE,cv2.GC_BGD,-1)

        cv2.imshow("show_img",self.show_img)

    def b_i_b_f(self):
        #붓소환
        self.label.setText("붓 소환")
        self.p_label.setText(f"{self.P_SIZE}")
        cv2.setMouseCallback("show_img",self.dw_b)
    
    def c_b_f(self):
        #오리기
        backgr=np.zeros((1,65),np.float64)
        forgr=np.zeros((1,65),np.float64)
        cv2.grabCut(self.img,self.mask,None,backgr,forgr,5,cv2.GC_INIT_WITH_MASK)
        mask2 = np.where((self.mask==cv2.GC_BGD)|(self.mask==cv2.GC_PR_BGD),0,1).astype('uint8')
        self.cut_img=self.img*mask2[:,:,np.newaxis]
        cv2.imshow("cut_img",self.cut_img)

    def f_s_b_f(self):
        #파일 저장
        s_fname=QFileDialog.getSaveFileName(self,"파일 저장",'./')
        cv2.imwrite(s_fname[0],self.cut_img)

    def e_b_f(self):
        #종료
        cv2.destroyAllWindows()
        self.close()

class Mc_img(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('이미지 검출기')
        self.setGeometry(200,200,570,100)
       
        ck_in_b=QPushButton('검출 정보 등록',self)
        ck_in_n_b=QPushButton('검출 정보 확인',self)
        l_im_b=QPushButton('검출대상 영상 불러옴',self)
        mc_b=QPushButton('인식',self)
        quitButton=QPushButton('나가기',self)
        self.label=QLabel('환영합니다!',self)
        self.label1=QLabel(' ',self)
        
        ck_in_b.setGeometry(10,10,100,30)
        ck_in_n_b.setGeometry(110,10,100,30)
        l_im_b.setGeometry(210,10,150,30)
        mc_b.setGeometry(360,10,100,30)
        quitButton.setGeometry(460,10,100,30)
        self.label.setGeometry(10,40,550,30)
        self.label1.setGeometry(10,70,550,30)
        
        ck_in_b.clicked.connect(self.ck_in_f)
        ck_in_n_b.clicked.connect(self.ck_in_n_f)
        l_im_b.clicked.connect(self.l_im_f) 
        mc_b.clicked.connect(self.mc_f)        
        quitButton.clicked.connect(self.quitFunction)

        self.signImgs=[]				# 표지판 모델 영상 저장
        
    def ck_in_f(self):
        self.label.clear()
        self.label.setText('검출 정보를 등록합니다.')

        l_fname=QFileDialog.getOpenFileName(self,"파일 로드",'./')
        self.ck_img=cv2.imread(l_fname[0])
        if self.ck_img is None:sys.exit("파일이 없습니다.")
        self.signImgs.append(self.ck_img)
        #cv2.imshow('ck_img',self.signImgs[-1])
        #cv2.imshow('ck_img',self.ck_img)  
        self.label1.setText(f'등록된 검출정보는 {len(self.signImgs)}개 입니다.')      
    def ck_in_n_f(self):
        self.label.clear()
        self.label.setText('등록된 검출 정보를 출력합니다.')
        for n,im in enumerate(self.signImgs):
            cv2.imshow(f'ck_img{n+1}',im) 


    def l_im_f(self):
        if self.signImgs==[]: 
            self.label.setText('먼저 검출 정보를 등록하세요.')
        else:
            fname=QFileDialog.getOpenFileName(self,'파일 로드','./')
            self.road_img=cv2.imread(fname[0])
            if self.road_img is None: sys.exit('파일을 찾을 수 없습니다.')  
    
            cv2.imshow('Road scene',self.road_img)  
        
    def mc_f(self):
        if self.road_img is None: 
            self.label.setText('검출대상 영상을 입력하세요.')
        else:
            sift=cv2.SIFT_create()
        
            KD=[] # 여러 표지판 영상의 키포인트와 기술자 저장
            for img in self.signImgs: 
                gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                KD.append(sift.detectAndCompute(gray,None))
                
            grayRoad=cv2.cvtColor(self.road_img,cv2.COLOR_BGR2GRAY) 
            road_kp,road_des=sift.detectAndCompute(grayRoad,None) 
                
            matcher=cv2.DescriptorMatcher_create(cv2.DescriptorMatcher_FLANNBASED)
            GM=[]			
            for sign_kp,sign_des in KD:
                knn_match=matcher.knnMatch(sign_des,road_des,2)
                T=0.7
                good_match=[]
                for nearest1,nearest2 in knn_match:
                    if (nearest1.distance/nearest2.distance)<T:
                        good_match.append(nearest1)
                GM.append(good_match)        
            
            best=GM.index(max(GM,key=len)) 
            
            if len(GM[best])<4:	
                self.label.setText('검출 대상이 없습니다.')  
            else:			
                sign_kp=KD[best][0]
                good_match=GM[best]
            
                points1=np.float32([sign_kp[gm.queryIdx].pt for gm in good_match])
                points2=np.float32([road_kp[gm.trainIdx].pt for gm in good_match])
                
                H,_=cv2.findHomography(points1,points2,cv2.RANSAC)
                
                h1,w1=self.signImgs[best].shape[0],self.signImgs[best].shape[1] 
                h2,w2=self.road_img.shape[0],self.road_img.shape[1] 
                
                box1=np.float32([[0,0],[0,h1-1],[w1-1,h1-1],[w1-1,0]]).reshape(4,1,2)
                box2=cv2.perspectiveTransform(box1,H)
                
                self.road_img=cv2.polylines(self.road_img,[np.int32(box2)],True,(0,255,0),4)
                
                img_match=np.empty((max(h1,h2),w1+w2,3),dtype=np.uint8)
                cv2.drawMatches(self.signImgs[best],sign_kp,self.road_img,road_kp,good_match,img_match,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                cv2.imshow('Matches and Homography',img_match)
                
                self.label.setText('대상이 검출 되었습니다.')                         
                      
    def quitFunction(self):
        cv2.destroyAllWindows()        
        self.close()
                

app = QApplication(sys.argv)
m_win=c_n()
m_win.show()
app.exec_()

app2 = QApplication(sys.argv)
m_win2=Cut_bk_img()
m_win2.show()
app2.exec_()

app3=QApplication(sys.argv) 
m_win3=Mc_img() 
m_win3.show()
app3.exec_()