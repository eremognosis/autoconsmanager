import numpy as np
########### global constants
mu = np.float64(398600.4418)         # km^3/s^2   ####Earth’s standard gravitational parameter
J2 = np.float64(0.00108263) ### unitless
RE = np.float64(6378.137) ### km radius of earth

eps = 1e-9  ####### softening (no zerodivision)
I_sp =  np.float64(300.00)
g0 = np.float64(0.00980665) # km/s^2


###########################
# We are defining a state object to be consiusting of [x,y,z, vx,vy,vz] np array
#####################

##########################
##  d^2r/dt^2 = - \mu/|r|^3 + a
############################
class physicsengine:
    def __init__(self, num_satellites, num_debris):
        self.num_sats = num_satellites  ####number of satelitres
        self.num_debris = num_debris  ###### number of debir
        self.total_objects = num_satellites + num_debris
        self.state_matrix = np.zeros((self.total_objects, 6), dtype=np.float64)
        
        ####
        self.ids = np.empty(self.total_objects, dtype=object)
        
        self.idtoind = {}
        self.issate = np.zeros(self.total_objects, dtype=bool)
        
        ### kind of metadata
        self.masses = np.ones(self.total_objects, dtype=np.float64) * 550.0 
        self.cooldowns = np.zeros(self.total_objects, dtype=np.float64)
        
        
    def calcderiv(self, state : np.ndarray) -> np.ndarray:
        """
        $(r,v) ->(v,a)$
        """
        
        
                
        pos = state[:, 0:3]
        vel = state[:, 3:6]
        
        r = np.linalg.norm(pos, axis=1)[:, np.newaxis]  ### calculate the norms i.e. dist of vectores and convert the at into coliumn matrix for consistency 
        
    
        ##### g = -\mu/r^3 x
        a_grav = -(mu)/(r**3 + eps) * pos
        
        
        coeff = (1.5 *mu* J2 * RE**2)/(r**5)  #### technically scalar so a column mat here

        # we need z to be N,1
        
        z = pos[:, 2][:, np.newaxis]
        z_thing = 5*(z**2)/(r**2)
        
        
        p_vec = np.empty_like(pos) 
        
        p_vec[:,0:2] = pos[:,0:2]*(z_thing-1)
        p_vec[:,2:3] = pos[:,2:3]*(z_thing-3)
        
        a_j2 = coeff * p_vec
        
        
        a_total = a_grav + a_j2
        
        
        der = np.hstack((vel, a_total))
        
        return der
    
    
    
    def onestep(self, dt):
        state = self.state_matrix
        
        
        
        k1 = self.calcderiv(state)
        k2 = self.calcderiv(state + 0.5 * dt * k1)
        k3 = self.calcderiv(state + 0.5 * dt * k2)
        k4 = self.calcderiv(state + dt * k3)
        ### we are using Rk4
        #####
        ####################
        
        self.state_matrix +=   (dt/6.0) * (k1+2*k2+2*k3+k4) 
        self.cooldowns = np.maximum(0, self.cooldowns-dt)
        
    # def delm(self, )
    def burn(self, i,  delv):
        ##### 15 m/s 0.015 km/s
        if (np.linalg.norm(delv) <= 0.015) and (self.cooldowns[i]<=0):
            self.state_matrix[i, 3:6]  += delv
            self.cooldowns[i] =600
            powerthing = (-np.linalg.norm(delv))/(I_sp*g0)
            delm = self.masses[i]*(1-np.exp(powerthing))
            if (self.masses[i] - delm >= 500.0):
                self.masses[i] -= delm
                
    def move(self, bigt): 
        ### to reduceb thje erron
        ## we divide in 5 second chunks
        n = int(bigt//5)   # --> int othwer siw range gives error
        rem = bigt -(n*5.0)
        for i in range(n):
            self.onestep(5.0)
        self.onestep(rem)
        
        
        
        
    def ingest(self, payload):
        objects = payload.get("objects", [])
        
        for obj in objects:
            oid = obj["id"]
            
            ############for new things
            if oid not in self.idtoind:
                idx = len(self.idtoind)
                if idx >= self.total_objects:
                    continue #### its says constaret
                
                self.idtoind[oid] = idx
                self.ids[idx] = oid
                self.issate[idx] = (obj["type"] == "SAT") # for now keey "SAT"$#$$$$$#######
            
            idx = self.idtoind[oid]
            self.state_matrix[idx, 0] = obj["r"]["x"]
            self.state_matrix[idx, 1] = obj["r"]["y"]
            self.state_matrix[idx, 2] = obj["r"]["z"]
            self.state_matrix[idx, 3] = obj["v"]["x"]
            self.state_matrix[idx, 4] = obj["v"]["y"]
            self.state_matrix[idx, 5] = obj["v"]["z"]
            
            
            
            
    def rtntoeci(self, i):
        ### its to convert the rtn time to earth centre
        r = self.state_matrix[i, 0:3]
        v = self.state_matrix[i, 3:6] ## obv
        
        rn = np.linalg.norm(r)
        R = r/(rn + eps)  ## rn cant be zero anywya
        
        L = np.cross(r, v)
        Ln = np.linalg.norm(L)
        
        N = L/(Ln+eps)

        T = np.cross(N,R)
        
        return np.column_stack((R,T,N))         ##### apparently i did an error by using just stack earlier
    
    
    def burnrtn(self,i, delvrtn):
        ## input giben in rtn
        rm = self.rtntoeci(i)
        delv = rm@ np.array(delvrtn)
        
        self.burn(i, delv)
        
    