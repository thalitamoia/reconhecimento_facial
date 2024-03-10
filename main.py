import cv2 
import mediapipe as mp

# inicializar OpenCV e Mediapipe
webcam = cv2.VideoCapture(0)
solucao_reconhecimento_rosto = mp.solutions.face_detection
reconhecedor_de_rostos = solucao_reconhecimento_rosto.FaceDetection()
desenho = mp.solutions.drawing_utils

while True:
   #ler as informações da webcam
   verificador, frame = webcam.read()
   if not verificador:
        break
   #reconhecer os rostos na imagem
   lista_rostos = reconhecedor_de_rostos.process(frame)
   if lista_rostos.detections:
       for deteccao in lista_rostos.detections:
           # desenhar retângulo ao redor do rosto
           desenho.draw_detection(frame, deteccao)

   # exibir a imagem com as detecções
   cv2.imshow('Faces', frame)
   
   # para quando a tecla ESC (27) for pressionada
   if cv2.waitKey(30) & 0xFF == 27:
        break

# liberar recursos
webcam.release()
cv2.destroyAllWindows()
