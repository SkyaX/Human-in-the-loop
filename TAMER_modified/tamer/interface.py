import os
import pygame
import speech_recognition as sr
from gtts import gTTS
import sys

import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
from pynput.keyboard import Controller

class Interface:
	""" Pygame interface for training TAMER """

	def __init__(self, action_map):
		self.action_map = action_map
		pygame.init()
		self.font = pygame.font.Font("freesansbold.ttf", 32)

		# set position of pygame window (so it doesn't overlap with gym)
		os.environ["SDL_VIDEO_WINDOW_POS"] = "1000,100"
		os.environ["SDL_VIDEO_CENTERED"] = "0"

		self.screen = pygame.display.set_mode((200, 100))
		area = self.screen.fill((0, 0, 0))
		pygame.display.update(area)

		self.cap = cv2.VideoCapture(0)
		self.cap.set(3,1280)
		self.cap.set(4,720)

		self.Hdetector = HandDetector(detectionCon = 0.8)
		
	"""def speak(audioString):
		tts = gTTS(text=audioString, lang='en')
		tts.save("audio.mp3")
		audio = pygame.mixer.Sound("audio.mp3")
		audio.play()"""

	def __del__(self):
		self.disp.cap.release()
		
	def recordAudio(self):
		# Record Audio
		print('record_audio')
		r = sr.Recognizer()
		with sr.Microphone() as source:
			r.adjust_for_ambient_noise(source)
			audio = r.listen(source)
	
		# Speech recognition using Google Speech Recognition
		data = ""
		try:
			data = r.recognize_google(audio)

		except sr.UnknownValueError:
			pass
		except sr.RequestError as e:
			pass
	
		return data
		
	def get_vocal_feedback(self):
		"""
		Get human input. 'W' key for positive, 'A' key for negative.
		Returns: scalar reward (1 for positive, -1 for negative)
		"""
		reward = 0
		area = None

		data = self.recordAudio()
		print(data)

		if data == "true":
			area = self.screen.fill((0, 255, 0))
			reward = 1

		elif data == "false":
			area = self.screen.fill((255, 0, 0))
			reward = -1

		pygame.display.update(area)
		return reward

	def get_visual_feedback(self):
		"""
		Get human input. 'W' key for positive, 'A' key for negative.
		Returns: scalar reward (1 for positive, -1 for negative)
		"""
		success, img = self.cap.read()
		img = cv2.flip(img,1)
		hands, img = self.Hdetector.findHands(img, draw=True, flipType=False)

		if hands :
			# 1
			hand1 = hands[0]
			lmList1 = hand1["lmList"] # List of 21 landmarks points
			bbox1 = hand1["bbox"] # Bounding box info : x, y, w, h
			centerPoint1 = hand1["center"] # center of the hand : cx, cy
			handType1 = hand1["type"] # Left or Right

			# print(lmList1, len(lmList1))

			fingers1 = self.Hdetector.fingersUp(hand1)
			# length, img = Hdetector.findDistance(lmList1[4], lmList1[8], img)
			Fing = self.Hdetector.fingersUp(hand1)

			print(Fing, fingers1)
			# print(centerPoint1)
			# print(handType1)

			if len(hands) == 2 :
				# 2
				hand2 = hands[1]
				lmList2 = hand2["lmList"] # List of 21 landmarks points
				bbox2 = hand2["bbox"] # Bounding box info : x, y, w, h
				centerPoint2 = hand2["center"] # center of the hand : cx, cy
				handType2 = hand2["type"] # Left or Right

				fingers2 = self.Hdetector.fingersUp(hand2)

				print(fingers1, fingers2)

			if fingers1[1] == 1:
				print('REWARD 1')
				area = self.screen.fill((0, 255, 0))
				reward = 1
				return reward
			else:
				print('REWARD -1')
				area = self.screen.fill((255, 0, 0))
				reward = -1
				return reward
		else:
			print('REWARD 0')
			area = self.screen.fill((0, 0, 255))
			reward = 0
			return reward

	def get_scalar_feedback(self):
		"""
		Get human input. 'W' key for positive, 'A' key for negative.
		Returns: scalar reward (1 for positive, -1 for negative)
		"""
		reward = 0
		area = None
		for event in pygame.event.get():
			if event.type == pygame.KEYDOWN:
				if event.key == pygame.K_w:
					area = self.screen.fill((0, 255, 0))
					reward = 1
					break
				elif event.key == pygame.K_a:
					area = self.screen.fill((255, 0, 0))
					reward = -1
					break
		pygame.display.update(area)
		return reward

	def show_action(self, action):
		"""
		Show agent's action on pygame screen
		Args:
			action: numerical action (for MountainCar environment only currently)
		"""
		area = self.screen.fill((0, 0, 0))
		pygame.display.update(area)
		text = self.font.render(self.action_map[action], True, (255, 255, 255))
		text_rect = text.get_rect()
		text_rect.center = (100, 50)
		area = self.screen.blit(text, text_rect)
		pygame.display.update(area)
