import numpy as np
import sys
import random as rd
from statistics import mean, stdev
from ast import literal_eval

player = sys.argv[1]

class Q_Agent():

	def __init__(self, num_feats):
		self.num_feats = num_feats
		self.weights = np.random.rand(num_feats)
		self.l_r = 0.01
		self.disc = 0.9
		self.reg_const = 0.01

	def extractFeatures(self,state, action):
		
		feats = np.zeros(self.num_feats)
		board, piece = state

		# no of filled spots
		np.append(feats,np.count_nonzero(board))

		def numOf3x3Spaces(board):
			b = 0
			count = 0
			for i in range(len(board)-3):
				for j in range(len(board[0]-3)):
					if np.count_nonzero(board[i:i+3,j:j+3])==0:
						b = 1
						count += 1
			return count, b

		#num of places for 3x3 block
		count, boo = numOf3x3Spaces(board)
		np.append(feats,count)
		np.append(feats,boo)
		
		#no of pieces in rows, and indicator for cleared row
		for i in range(len(board)):
			count = np.count_nonzero(board[i])
			if count==0:
				np.append(feats,1)
			else:
				np.append(feats,0)
			np.append(feats,count)

		#no of pieces in cols, and indicator for cleared col
		for i in range(len(board[0])):
			count = np.count_nonzero(board[:,i])
			if count==0:
				np.append(feats,1)
			else:
				np.append(feats,0)
			np.append(feats,count)

		# weighted sum of filled spots, with higher weights to the edges

		# almost filled columns and rows

		# no of empty rows and columns

		for i in range(len(board)):
			for j in range(len(board[0])):
				np.append(feats,1 if board[i,j] else 0)
		
		return feats


	def chooseAction(self,state,actions):

		#action = (x,y)
		best_q = None
		best_action = None

		for action in actions:
			feats = agent.extractFeatures(state,action)
			q = np.dot(self.weights,feats)
			if best_action is None or q>best_q:
				best_action = action
				best_q = q
		return best_action

	def updateWeights(self,state, action, reward, next_state):
		feats = agent.extractFeatures(state,action)
		board, piece = state
		next_board, next_piece = next_state
		n_s_poss_actions = availableIndices(next_board, next_piece)
		q_opt = np.dot(self.weights,feats)
		if len(n_s_poss_actions)==0:
			v_opt = 0
		else:
			v_opt = max([np.dot(self.weights,agent.extractFeatures(next_state,new_action)) for new_action in n_s_poss_actions])
		self.weights -= self.l_r*((q_opt-(reward+self.disc*v_opt))*feats+self.reg_const*self.weights)

if player == 'q':
	agent = Q_Agent(143)


def availableIndices(board, piece):
	indices = []
	arr = np.array(pieces[piece][0])
	h, w = arr.shape
	for i in range(len(board)-h+1):
		for j in range(len(board[0])-w+1):
			if np.all(np.logical_not(np.logical_and(board[i:i+h,j:j+w],arr))): #if all of the entries are false:
				indices.append((i,j))
	return indices

pieces = {

	'5-block-h': ([[1,1,1,1,1]], 5, 1./19),

	'4-block-h': ([[1,1,1,1]], 4, 1./19),

	'3-block-h': ([[1,1,1]], 3, 1./19),

	'2-block-h': ([[1,1]], 2, 1./19),

	'1-block-h': ([[1]], 1, 1./19),

	'5-block-v': ([[1],[1],[1],[1],[1]], 5, 1./19),

	'4-block-v': ([[1],[1],[1],[1]], 4, 1./19),

	'3-block-v': ([[1],[1],[1]], 3, 1./19),

	'2-block-v': ([[1],[1]], 2, 1./19),

	'l-2-a': ([[1,0],[1,1]], 3, 1./19),

	'l-2-b': ([[0,1],[1,1]], 3, 1./19),

	'l-2-c': ([[1,1],[0,1]], 3, 1./19),

	'l-2-d': ([[1,1],[1,0]], 3, 1./19),

	'2-square': ([[1,1],[1,1]], 4, 1./19),

	'3-square': ([[1,1,1],[1,1,1],[1,1,1]], 9, 1./19),

	'l-3-a': ([[1,0,0],[1,0,0],[1,1,1]], 5, 1./19),

	'l-3-b': ([[1,1,1],[0,0,1],[0,0,1]], 5, 1./19),

	'l-3-c': ([[0,0,1],[0,0,1],[1,1,1]], 5, 1./19),

	'l-3-d': ([[1,1,1],[1,0,0],[1,0,0]], 5, 1./19)

}

def playGame():

	board = np.full((10,10), False)
	score = 0

	piecelist = list(pieces.items())
	piecenames = [piece[0] for piece in piecelist]
	pieceprobs = [piece[1][2] for piece in piecelist]

	piece = np.random.choice(piecenames,p=pieceprobs)

	while True:
		#print(piece)

		indices = availableIndices(board,piece)
		if len(indices)==0:
			break

		arr = np.array(pieces[piece][0])
		h, w = arr.shape
		#ask user for entry index, until valid index is input

		x,y = pickMove(board, piece, indices)

		new_board = np.copy(board)

		#place piece
		for i in range(h):
			for j in range(w):
				new_board[x+i,y+j] = arr[i,j]

		#eliminate rows/cols

		cols_to_elim = []
		rows_to_elim = []

		for i in range(len(new_board)):
			if all(new_board[i]):
				rows_to_elim.append(i)

		for j in range(len(new_board[0])):
			if all(new_board[:,j]):
				cols_to_elim.append(j)

		for col in cols_to_elim:
			new_board[:,col] = np.full(len(new_board), False)

		for row in rows_to_elim:
			new_board[row] = np.full(len(new_board[0]), False)

		#update score, randomly pick new piece

		new_score = score + (5+5*(len(rows_to_elim)+len(cols_to_elim)))*(len(rows_to_elim)+len(cols_to_elim))+pieces[piece][1]

		#name = input("press enter for next step")

		new_piece = np.random.choice(piecenames,p=pieceprobs)

		agent.updateWeights((board,piece), (x,y), new_score - score, (new_board,new_piece))

		score = new_score
		piece = new_piece
		board = new_board


	return score

def pickMove(board, piece, indices):
	if player == 'random':
		return rd.choice(indices)

	if player=='q':
		state = (board,piece)
		return agent.chooseAction(state,indices)

	else:
		print(board)
		print(pieces[piece][0])
		move = literal_eval(input("enter index (x,y): "))
		while move not in indices:
			move = literal_eval(input("enter index (x,y): "))
		return move


if player == 'random':
	scores = []
	counter=0
	while counter<1000:
		counter+=1
		scores.append(playGame())
	print(mean(scores))
	print(stdev(scores))

elif player == 'q':
	s = 0
	for t in range(100):
		sc = playGame()
		print(sc)
		s += sc
		#print(agent.weights)
	print(s/100)

else:
	print(playGame())

