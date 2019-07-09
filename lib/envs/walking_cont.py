import numpy as np
import sys
import random
from gym.envs.toy_text import discrete
import random

UP = 0
RIGHT = 1
DOWN = 2
LEFT =3

class GridworldEnv(discrete.DiscreteEnv):
    metadata = {'render.modes':['human', 'ansi']}
    
    
    def _limit_coordinates(self, coord):
        coord[0] = min(coord[0], self.shape[0] -1)
        coord[0] = max(coord[0],0)
        
        coord[1] = min(coord[1], self.shape[1] -1)
        coord[1] = max(coord[1], 0)
        
        return coord
    
    def _calculate_transition_prob(self, current, delta):
        nS = np.prod(self.shape)
        new_position = np.array(current) + np.array(delta)
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        
        
        #for i in range(0,3):
        #    if new_state==self.target[i]:
        #        self.t[i]=1
        reward = -3.0 + sum(self.t)

        is_done = all(tar ==1 for tar in self.t)
        #print(self.t)

        return [(1.0, new_state, reward, is_done)]

    def reset(self):
        self.t = [0,0,0]
        self.reward=0
        return self.s

    def step(self, a):
        nS = np.prod(self.shape)
        transitions = self.P[self.s][a]
        #i = categorical_sample([t[0] for t in transitions], self.np_random)
        p,s,r,d = transitions[0]
        '''
        self.target[0] = random.randint(-1,nS-1)
        self.target[1] = random.randint(-1,nS-1)
        self.target[2] = random.randint(-1,nS-1)
        '''
        self.target[0] = nS-1 #if self.target[0]==nS-1 else self.target[0]+1
        self.target[1] = 0 #self.target[1]=7 if self.target[1]==14 else  self.target[1]+1 
        self.target[2] = 0 #if self.target[2]==0 else self.target[2]-1
        '''
        self.target[0] = nS-1 if self.target[0]==nS-1 else self.target[0]+1
        self.target[1] = 17 if self.target[1]+32>nS-1 else  self.target[1]+32 
        self.target[2] = 0 if self.target[2]==0 else self.target[2]-1
        '''
        #print(self.target)
        for i in range(0,3):
            if s==self.target[i]:
                self.t[i]=1
        d = all(tar ==1 for tar in self.t)
        #r = 0.0 if d else -1.0
        r = -3.0 + sum(self.t)
        #d = all(tar ==1 for tar in self.t)
        self.s = s
        self.lastaction = a
        return (s, r, d, {"prob" : p})
    
    def __init__(self):
        self.shape = (4,2)
        
        nS = np.prod(self.shape) #number of states
        nA = 4 #number of actions

        self.t = [0,0,0]

        self.target = [1,15,nS-1]
        
        #calculate transistion probabilities
        P = {}
        for s in range(nS):
            position = np.unravel_index(s, self.shape)
            P[s] = {a : [] for a in range (nA)}
            
            P[s][UP] = self._calculate_transition_prob(position, [-1,0])
            P[s][RIGHT] = self._calculate_transition_prob(position, [0,1])
            P[s][DOWN] = self._calculate_transition_prob(position,[1,0])
            P[s][LEFT] = self._calculate_transition_prob(position, [0,-1])
            
        #we always start in state (3,0)
        isd = np.zeros(nS)
        isd[np.ravel_multi_index((3,0) , self.shape)] = 1.0
        
        super(GridworldEnv, self).__init__(nS,nA,P,isd)
        
    def render(self,mode='human', close=False):
        self._render(mode, close)
        
    def _render(self, mode='human', close=False):
        if close:
            return
            

        outfile = StringIO() if mode=='ansi' else sys.stdout
        print(self.target)
        
        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            if self.s == s:
                output = " x "
            elif any(tar ==s for tar in self.target):
                output = " T "
            else:
                output = " o "
                
            if position[1] == 0:
                output = output.lstrip()
            if position[1] == self.shape[1] -1:
                output = output.rstrip()
                output += "\n"
                
            outfile.write(output)
        outfile.write("\n")
