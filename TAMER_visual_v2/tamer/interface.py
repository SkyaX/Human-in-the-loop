import os
import pygame
import sys
import time
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
		self.cap.set(3,480)
		self.cap.set(4,240)

		self.Hdetector = HandDetector(detectionCon = 0.8)


	def get_visual_feedback(self):
		"""
		Get human input via webcam. Two hands closed and visible for positive, Two hands opened and visible for negative.
		Returns: scalar reward (1 for positive, -1 for negative)
		"""

		time.sleep(0.2)
		success, img = self.cap.read()
		img = cv2.flip(img,1)
		hands, img = self.Hdetector.findHands(img, draw=True, flipType=False)
		
		while len(hands) != 2 :
			print(f"\n\tNot enough hands shown, we need your two hands to register the reward, you have 5 seconds to place yourself !\n")
			i = 0
			while i!=5 :
				i+=1
				print(f"{5-i} !")
				time.sleep(0.5)

			success, img = self.cap.read()
			img = cv2.flip(img,1)
			hands, img = self.Hdetector.findHands(img, draw=True, flipType=False)

		# 1
		hand1 = hands[0]
		fingers1 = self.Hdetector.fingersUp(hand1)
		# 2
		hand2 = hands[1]
		fingers2 = self.Hdetector.fingersUp(hand2)

		if fingers1[1] == 0 & fingers2[1] == 0:
			print('REWARD 1')
			area = self.screen.fill((0, 255, 0))
			reward = 1
			return reward
		elif fingers1[1] == 1 & fingers2[1] == 1:
			print('REWARD -1')
			area = self.screen.fill((255, 0, 0))
			reward = -1
			return reward
		else : # (fingers1[1] == 1 & fingers2[1] == 0) | (fingers1[1] == 0 & fingers2[1] == 1) :
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
