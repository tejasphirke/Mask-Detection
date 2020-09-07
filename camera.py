import cv2
import boto3 # to communicate with aws services
import datetime
import requests
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
ds_factor=0.6

count=0

class VideoCamera(object):    
    def __init__(self):
        self.video = cv2.VideoCapture(0)
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        #count=0
        global count
        success, image = self.video.read()
        is_success, im_buf_arr = cv2.imencode(".jpg", image)
        image1 = im_buf_arr.tobytes()
        client=boto3.client('rekognition',
                        aws_access_key_id="ASIATHTSQNSRAYECSN7R",
                        aws_secret_access_key="3pHW0p5mNfZgWsKMp/Pwtk4SeH17Uyc/4VymdQhk",
                        aws_session_token="FwoGZXIvYXdzEBYaDI55GWcZcwewKhU/CCLMAXFYiyM+XBkBotMve1M1MreaaR/k9l0fk4k2wsYBAoSREOckwaZWHikbmhC9Xh8OM36palYsMXNK2VdJszew2W7d2h8JHrRGjs+G+1SdOf7iLpsyXQlbZF8ENK0l9WLFclq1XQ1xdzAe5o/gYB7IjFKRM+vD4PXMka+yglTRBK4C20M7KH1MYxqnEO24hsd9bhbg9YndosOAIXUbWOyNEM92XOIGC5xX4UWYIZn//wfoQqiViDE1Ygc1vX76ZoWWvROE4dEXjoKqkTBnuCj4/Nb6BTItD4Yck/1gwkAsPdryZZVCTHy+gVNy0jtYAImndvYkkImEraJvJwSO36OEzhc2",
                        region_name='us-east-1')
        response = client.detect_custom_labels(
        ProjectVersionArn='arn:aws:rekognition:us-east-1:222504709282:project/mask-detection-using-aws/version/mask-detection-using-aws.2020-09-05T20.57.09/1599319630473',Image={
            'Bytes':image1})
        print(response['CustomLabels'])
        
        if not len(response['CustomLabels']):
            count=count+1
            date = str(datetime.datetime.now()).split(" ")[0]
            #print(date)
            url = "https://0letodqt4m.execute-api.us-east-1.amazonaws.com/apiMaskCount/maskcount?date="+date+"&count="+str(count)
            resp = requests.get(url)
            f = open("countfile.txt", "w")
            f.write(str(count))
            f.close()
            #print(count)

        image=cv2.resize(image,None,fx=ds_factor,fy=ds_factor,interpolation=cv2.INTER_AREA)
        
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        face_rects=face_cascade.detectMultiScale(gray,1.3,5)
        for (x,y,w,h) in face_rects:
        	cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        	break
        ret, jpeg = cv2.imencode('.jpg', image)
        #cv2.putText(image, text = str(count), org=(10,40), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(1,0,0))
        cv2.imshow('image',image)
        return jpeg.tobytes()
