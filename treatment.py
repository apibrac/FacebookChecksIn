import csv
import numpy as np



def extract_from(name,X,Y,dx=0,dy=0,transform_type=lambda x:x):
    '''extract too list of lines (with x and y at 0 and 1): those delimited by X and Y, those that are dx and/or dy further'''
    small,big=[],[]
    with open(name,newline='') as f :
        reader=csv.reader(f)
        head=reader.__next__()
        for line in reader:
            line=transform_type(*line)
            x,y=line[0:2]
            if x > X[0]-dx and x < X[1]+dx and y > Y[0]-dy and y < Y[1]+dy : # in the big square
                if x > X[0] and x < X[1] and y > Y[0] and y < Y[1] : #also in the small one
                    small.append(line)
                else :
                    big.append(line)
        return head,small,big
    
    
    
def extend_from(X,Y=None,dt=0,indice_temps=3):
    '''send a copy of X (and Y) lines that should be at the border following the time'''
    if not Y:
        Y=X
    outX,outY=[],[]
    for x,y in zip(X,Y):
        if x[indice_temps] < dt :
            cx=list(x)
            cx[indice_temps]+=1440
            outX.append(tuple(cx))
            outY.append(y)
        if x[indice_temps] > 1440-dt:
            cx=list(x)
            cx[indice_temps]-=1440
            outX.append(tuple(cx))
            outY.append(y)
    return outX,outY
            
    
def cut(X,Y=None,n=3):
    size=len(X)//n
    np.random.seed()
    indices = np.random.permutation(len(X))
    Xtr=[X[i] for i in indices[:-size]]
    Xte=[X[i] for i in indices[-size:]]
    if Y:
        Xtr=[Y[i] for i in indices[:-size]]
        Xte=[Y[i] for i in indices[-size:]]
    else:
        Ytr,Yte=None,None
    return Xtr,Xte,Ytr,Yte

def separate_column(table,col=5):
    ar=np.array(table)
    ind=list(range(len(table[0])))
    ind.remove(col)
    return ar[:,ind],np.ravel(ar[:,[col]]).astype('int64')



def rate(solution,proposal):
    '''rate the proposal following the solution with Mean Average Precision'''
    out=0
    for u,v in zip(solution,proposal):
        for i,vi in enumerate(v):
            if u==vi :
                out+=1/(i+1)
                break
    return out/len(solution)


def n_best(classifier,X,n=3):
    '''Give the three best predictions (in order) by classifier for each row of X'''
    proba=classifier.predict_proba(X)
    out=[]
    for p in proba :
        p=list(p)
        list_of_proba=sorted(p)
        solutions=[]
        for a in range(n):
            i=list_of_proba.pop()
            if i==0:
                break
            solutions.append(classifier.classes_[p.index(i)])
        out.append(tuple(solutions))
    return out

def test(classifier, X_test, y_test, n=3):
    '''rate the classifier with Mean Average Precision @ n'''
    return rate(y_test,n_best(classifier,X_test,n))


def cut_in(a,b):
    def cell(na,nb):
        return (10*na/a,10*(na+1)/a),(10*nb/b,10*(nb+1)/b)
    return cell