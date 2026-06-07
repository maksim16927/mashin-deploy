import cv2 #импортирование библиотеки cv2
import time #импортирование библиотеки time 
import numpy as np #импортирование библиотеки numpy

class StreamHandler:
	stream = None #поток
	width = 0 #ширина кадра
	height = 0 #высота кадра
	fps = 0 #количество кадров в секунду потока
	controlFrames = [] #список кадров для проверки правильности распознавания линий разметки
	controlDirection = [] #список кадров для проверки правильности распознавания направления
	template_straight = None #шаблон прямой стрелки
	template_right = None #шаблон стрелки для повора направо

	def __init__(self, stream): #инициализация 
		self.stream = stream #прием потока
		self.width = round(stream.get(cv2.CAP_PROP_FRAME_WIDTH)) #считывание ширины кадра
		self.height = round(stream.get(cv2.CAP_PROP_FRAME_HEIGHT)) #считывание высоты кадра
		self.fps = stream.get(cv2.CAP_PROP_FPS) #считывание fps потока
		self.uploadTemplates() #подгрузка шаблонов

	def getSize(self):
		return [self.width, self.height]

	def getState(self):
		return self.stream.isOpened()

	def contourProcess(self, contours): #вычисление сплошности разметки
		sumLeftZone = 0 
		sumRightZone = 0
		for contour in contours: #проход по всем контурам
			for dot_packed in contour: #проход по всем точкам контура
				#print(dot_packed[0][0])
				dot = []
				dot.append(int(dot_packed[0][0]))
				dot.append(int(dot_packed[0][1]))
				#print(dot)
				#проверка нахождения точки в нужной зоне
				if(dot[0] >= int(self.width*0.3) and dot[0] <= int(self.width*0.5)):
					sumLeftZone += 1
				elif(dot[0] >= int(self.width*0.75) and dot[0] <= int(self.width*0.95)):
					sumRightZone += 1


		#print("SumLeftZone: ", sumLeftZone)
		#print("SumRightZone: ", sumRightZone)
		#time.sleep(5)
		return [sumLeftZone, sumRightZone]

	def uploadTemplates(self): #подгрузка шаблонов
		template_straight = cv2.imread('straight.jpg') #загрузка картинки прямой стрелки
		template_right = cv2.imread('right.jpg') #загрузка картинки стрелки направо

		#размытие шаблонов
		template_straight = cv2.medianBlur(template_straight,5) 
		template_right = cv2.medianBlur(template_right,5)

		#пороговое преобразование шаблонов
		ret, template_straight = cv2.threshold(template_straight, 210, 255, 0)
		ret, template_right = cv2.threshold(template_right, 210, 255, 0)

		#преобразование цвета в черно-белый
		template_straight = cv2.cvtColor(template_straight, cv2.COLOR_BGR2GRAY)
		template_right = cv2.cvtColor(template_right, cv2.COLOR_BGR2GRAY)

		#гауссовское размытие шаблонов
		self.template_straight = cv2.GaussianBlur(template_straight, (7, 7), 1.5)
		self.template_right = cv2.GaussianBlur(template_right, (7, 7), 1.5)

	def findDirectionMarks(self): #вычисление направления
		dir_right = 0 #сумма значения правого направления
		dir_straight = 0 #сумма значения прямого направления
		dir_threshold = 20 #порог точного распознавания

		for proc_frame in self.controlDirection: #проход по всем кадрам в списке проверки
			#cv2.imshow('template_straight', template_straight)
			#cv2.imshow('template_right', template_right)
			#cv2.imshow('proc_frame_directions', proc_frame)

			#проверка правого направления
			w, h = self.template_right.shape[::-1]
			resRight = cv2.matchTemplate(proc_frame,self.template_right,cv2.TM_CCOEFF_NORMED) #совмещение шаблона и кадра
			threshold = 0.8 #порог распознавания
			loc = np.where( resRight >= threshold) #проверка на прохождение порога
			#print(len(loc))
			
			for pt in zip(*loc[::-1]): #проход по всем найденным стрелкам
				dir_right += 1 #при нахождении добавления значения в сумму
				#cv2.rectangle(proc_frame, pt, (pt[0] + w, pt[1] + h), (255,255,255), 2)
			
			w, h = self.template_straight.shape[::-1]
			resStraight = cv2.matchTemplate(proc_frame,self.template_straight,cv2.TM_CCOEFF_NORMED) #совмещение шаблона и кадра
			threshold = 0.4 #порог распознавания
			loc = np.where( resStraight >= threshold) #проверка на прохождение порога
			#print(len(loc))
			
			for pt in zip(*loc[::-1]): #проход по всем найденным стрелкам
				dir_straight += 1 #при нахождении добавления значения в сумму
				#cv2.rectangle(proc_frame, pt, (pt[0] + w, pt[1] + h), (255,255,255), 2)

			#остановка проверки при преждевременном прохождении порога
			if dir_straight > dir_threshold:
				break
			elif dir_right > dir_threshold:
				break

		#print(dir_straight)
		#вычисление направления
		#cv2.imshow('directions', self.controlDirection[0])
		if(dir_straight > dir_threshold):
			return "STRAIGHT"
		elif(dir_right > dir_threshold):
			return "RIGHT"
		else:
			return "NONE"



	def frameProccess(self, init_frame): #ф-ия обработки кадра
		init_frame = cv2.rectangle(init_frame,(0, self.height//2),(self.width, round((self.height//4)*2.8)),(0,0,255),2) #отрисовка прямоугольника, на котором происходит распознавание
		proc_frame = init_frame[self.height//2:round((self.height//4)*2.8), 0:self.width] #получение кадра для обработки


		#cv2.imshow("proc_frame1",proc_frame)
		proc_frame = cv2.medianBlur(proc_frame,5) #применение размытия
		#cv2.imshow("proc_frame2",proc_frame)
		ret, proc_frame = cv2.threshold(proc_frame, 210, 255, 0) #применение порогового преобразования
		#cv2.imshow("proc_frame3",proc_frame)
		proc_frame = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2GRAY) #перевод в черно белый цвет
		#cv2.imshow("proc_frame4",proc_frame)
		proc_frame = cv2.GaussianBlur(proc_frame, (7, 7), 1.5) #гауссовское преобразование
		direction_frame = proc_frame[:,int(self.width*0.5)-10:int(self.width*0.75) + 10] #получение кадра для распознавания направления
		#cv2.imshow("proc_frame5",proc_frame)
		proc_frame = cv2.Canny(proc_frame, 1, 50) #применение кенни функции
		
		#отрисовка прямоугольников распознаваемых зон на исходном кадре
		init_frame = cv2.rectangle(init_frame,(int(self.width*0.3), self.height//2),(int(self.width*0.5), round((self.height//4)*2.8)),(0,255,255),2)
		init_frame = cv2.rectangle(init_frame,(int(self.width*0.75), self.height//2),(int(self.width*0.95), round((self.height//4)*2.8)),(0,255,255),2)

		contours, hierarchy = cv2.findContours(proc_frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE) #получение контуров и их иерархии из обработонного кадра
		cv2.drawContours(init_frame[self.height//2:round((self.height//4)*2.8), 0:self.width], contours, -1, (255,0,0), 5, cv2.LINE_AA, hierarchy, 1 ) #отрисовка контуров на исходном кадре

		if len(self.controlFrames) != 10: #проверка количество кадрав в списках для проверки
			#добавление кадров в списки проверки
			self.controlDirection.append(direction_frame) 
			self.controlFrames.append(self.contourProcess(contours)) #добавление значения сплошности разметки
		else:
			#добавление кадров в списки и удаление старых
			self.controlFrames = self.controlFrames[1:]
			self.controlFrames.append(self.contourProcess(contours))
			self.controlDirection = self.controlDirection[1:]
			self.controlDirection.append(direction_frame)

			#вычисление среднего значения сплошности линий разметки правой и левой зоны
			avgLeftZone = 0
			avgRightZone = 0
			for i in range(10):
				avgLeftZone += self.controlFrames[i][0]
				avgRightZone += self.controlFrames[i][1]
			avgLeftZone /= 10
			avgRightZone /= 10

			direction = self.findDirectionMarks() #поиск стрелок направления
			init_frame = cv2.putText(init_frame, "Direction %s" % (direction), (20,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2) #отрисовка текста с направлением
			init_frame = cv2.putText(init_frame, "LeftZone: %d, RightZone: %d" % \
				(int(avgLeftZone), int(avgRightZone)), (20,20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2) #отрисовка текста с сплошностью линий разметки
			
			#вычисление типа линий разметки
			out = ['','']
			if avgLeftZone > 1000:
				out[0] = "Solid"
			else:
				out[0] = "Intermittent"
			if avgRightZone > 1000:
				out[1] = "Solid"
			else:
				out[1] = "Intermittent"

			init_frame = cv2.putText(init_frame, "LeftZone: %s, RightZone: %s" % \
				(out[0], out[1]), (80,230), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2) #отрисовка текста с типом линий разметки
		
		return [proc_frame, init_frame] #возврат исходного и обработанного кадра
		            
		            
	def startStream(self):
		try: #запуск с защитой от ошибок
		    while self.getState(): #продолжение работы, пока поток открыт
		        ret, init_frame = self.stream.read() #чтение кадра из потока
		        if ret: #при непустом кадре
		            proc_frame, init_frame = self.frameProccess(init_frame) #обработка кадра
		            cv2.imshow("proc_frame",proc_frame) #вывод обработонного 
		            cv2.imshow("frame",init_frame) #вывод исходного кадра
		        if cv2.waitKey(1) & 0xFF == ord('q'): #проверка нажатия кнопки q для выхода из программы
		            break #остановка программы
		        else:
		            time.sleep(1/self.fps) #пауза между кадрами

		except KeyboardInterrupt: #отслеживание нажатия ctrl+c для принудительной остановки программы
		    print("Принудительная остановка")
		    self.stream.release() #остановка потока
		    cv2.destroyAllWindows() #закрытие всех окон cv2

		else: #отслеживание других исключений и нормальной остановки
		    print("Остановка видеопотока")
		    self.stream.release() #остановка потока
		    cv2.destroyAllWindows() #закрытие всех окон cv2
