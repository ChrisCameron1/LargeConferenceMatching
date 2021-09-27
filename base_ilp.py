import numpy as np

class BaseILP:
    def __init__(self):
        self.objective = Objective()
        self.constraints = Constraints()
        self.bounds = Bounds()
        self.general = General()
        self.binary = Binary()
        
    def reset(self):
        #easy hacky reset. ideally should delete the variables
        self.objective = Objective()
        self.constraints = Constraints()
        self.bounds = Bounds()
        self.general = General()
        self.binary = Binary()

    
    def write_to_file(self,file_name):
        with open(file_name,'w') as fh:
            print(self.objective.to_string(),file=fh)
            print(self.constraints.to_string(),file=fh)
            print(self.bounds.to_string(),file=fh)
            print(self.general.to_string(),file=fh)
            print(self.binary.to_string(),file=fh)
            print("End",file=fh)


        
class Equation:
    def __init__(self,eqn_type,name,var_coefs,oper,rhs):
        self.eqn_type = eqn_type #obj, or cons
        self.name = name # used in the LP file. 
        self.var_coefs = var_coefs # list of (variable name,coef) tuple
        self.oper = oper # operator in between . valid entries: = , >= , <=, > , <
        self.rhs = rhs #rhs. using default of 0.
        
    def to_string(self):
        #if self.eqn_type == 'obj':
            
        if self.name != '' and self.eqn_type == 'cons':
            prefix = '{}: '.format(self.name)
            #ans = '{}: {}'.format(self.name,lhs)
        else:
            prefix = ''
            #ans = lhs
        #
        lhs = ' '.join([ '{:+} {}'.format(np.round_(x[1],decimals=3),x[0]) for x in self.var_coefs])
        if self.eqn_type == 'obj':
            suffix = ''
        else:
            suffix = ' {} {}'.format(self.oper,self.rhs)
            
        return '{}{}{}'.format(prefix, lhs, suffix)

    
class Objective:
    def __init__(self):
        self.objectives = []
    
    def add(self,eqn):
        if isinstance(eqn,list):
            self.objectives.extend(eqn)
        else:
            self.objectives.append(eqn)
        
    def to_string(self):
        if len(self.objectives) == 0:
            return ''
        
        return 'MAXIMIZE\nobj: {}'.format(
            '\n'.join([eqn.to_string() for eqn in self.objectives]))
    
    
    
class Constraints:
    def __init__(self):
        self.constraints = []
        
    def add(self,eqn):
        if isinstance(eqn,list):
            self.constraints.extend(eqn)
        else:
            self.constraints.append(eqn)
        
    def to_string(self):
        if len(self.constraints) == 0:
            return ''
        
        return '\nSUBJECT TO\n {}'.format('\n'.join([eqn.to_string() for eqn in self.constraints]))
        
        
class General:
    def __init__(self):
        self.vars = []
        
    def add(self,var):
        if isinstance(var,list):
            self.vars.extend(var)
        else:
            self.vars.append(var)
    
    def to_string(self):
        if len(self.vars) == 0:
            return ''
        
        return '\nGENERAL\n{}'.format('\n'.join(self.vars))
        
class Binary:
    def __init__(self):
        self.vars = []
        
    def add(self,var):
        if isinstance(var,list):
            self.vars.extend(var)
        else:
            self.vars.append(var)
    
    def to_string(self):
        if len(self.vars) == 0:
            return ''
        
        return '\nBINARY\n{}'.format('\n'.join(self.vars)) # CC: Changed to have new lines between every var
        

class Bounds:
    def __init__(self):
        self.vars = []
        self.lower = []
        self.upper = []
   
    def __len__(self):
        return len(self.vars)

    def add(self,var,low = None,up = None):
        if isinstance(var,list):
            self.vars.extend(var)
            if low is None:
                low = [None]*len(var)
            self.lower.extend(low)
            if up is None:
                up = [None]*len(var)
            self.upper.extend(up)
        else:
            self.vars.append(var)
            self.lower.append(low)
            self.upper.append(up)
        
    def to_string(self):
        if len(self.vars) == 0:
            return ''
        
        bds = []
        for (l,v,u) in zip(self.lower,self.vars,self.upper):
            if l is None and u is None:
                raise Exception("Both lower and upper bounds cannot be None. Var: {}".format(v))
            if l is None:
                bds.append('{} <= {}'.format(v,u))
            elif u is None:
                bds.append('{} <= {}'.format(l,v))
            else:
                bds.append('{} <= {} <= {}'.format(l,v,u))
        return 'BOUNDS\n{}'.format('\n'.join(bds))
    
