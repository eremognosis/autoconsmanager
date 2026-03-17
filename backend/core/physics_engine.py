import numpy as np
########### global constants
mu = np.float64(398600.4418)         # km^3/s^2   ####Earth’s standard gravitational parameter
J2 = np.float64(0.00108263) ### unitless
RE = np.float64(6378.137) ### km radius of earth

eps = 1e-9  ####### softening (no zerodivision)



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