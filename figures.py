
import collections
import numpy as np
from sklearn.cross_validation import KFold, cross_val_predict
from matplotlib import pyplot as plt







class Testor():
    def __init__(this,X,y,cross_validation=3,grouping=20):
        this.X=X
        this.y=y
        this.Ncv=cross_validation
        this.grouping=grouping
        this.cvi=KFold(len(X), n_folds=cross_validation, shuffle=True, random_state=None)
        
        
    def testNrate(this,classifier,weights=None,output_file_name='testNrate',info=''):
        predicted = cross_val_predict(classifier, this.X*weights, this.y, cv=this.cvi, n_jobs=1, verbose=0, fit_params=None, pre_dispatch='2*n_jobs')
        right=this.y==predicted
        fig = plt.figure(figsize=(40, 40))
        fig.suptitle("Accuracy: {0:.2f}%. Cross validation done with {1} segments.\n".format(sum(right)/len(right)*100,this.Ncv)+info)#, fontsize=60)
        ax1 = fig.add_subplot(221,aspect=1)
        ax2 = fig.add_subplot(222,aspect=1)
        X=this.X
        good=[(float(a),float(b)) for a,b,c in zip(X[:,[0]],X[:,[1]],right) if c]
        wrong=[(float(a),float(b)) for a,b,c in zip(X[:,[0]],X[:,[1]],right) if not c]
        ax1.scatter(*zip(*good),linewidths=0)
        ax1.set_title('Good results')
        ax2.scatter(*zip(*wrong),linewidths=0)
        ax2.set_title('Bad results')
        ax3 = fig.add_subplot(413)
        ax4 = fig.add_subplot(414)
        hists_plot(ax3,ax4,this.y,right,this.grouping)
        fig.savefig('data/'+output_file_name+'.png')
        return fig



























def hist_plot(ax,y,right,until=None,group=20):
    numberTot=collections.defaultdict(int)
    numberFou=collections.defaultdict(int)
    for real_v, found in zip(y,right):
        numberTot[real_v]+=1
        if found:
            numberFou[real_v]+=1
    mylist=[]
    for i in numberTot:
        mylist.append((numberTot[i],numberFou[i]))
    mylist.sort(reverse=True)
    if not until:
        until=len(mylist)
    ax.bar(range(until),list(zip(*mylist))[0][:until],linewidth=0,width=1)
    ax.bar(range(until),list(zip(*mylist))[1][:until],linewidth=0,color='r',width=1)
    axt=ax.twinx()
    ratio=[r/e for e,r in mylist][:until]
    axt.plot(range(until),ratio,color='g')
    average=collections.defaultdict(list)
    for i,v in enumerate(ratio):
        average[i//group].append(v)
    ratio2=[]
    for key in average:
        ratio2.append(sum(average[key])/len(average[key]))
    pos=[(1/2+n)*group for n in range(len(ratio2))]
    axt.plot(pos,ratio2,color='r')
    
    
    
def hists_plot(ax,ax2,y,right,group=20):
    numberTot=collections.defaultdict(int)
    numberFou=collections.defaultdict(int)
    average=collections.defaultdict(list)
    for real_v, found in zip(y,right):
        numberTot[real_v]+=1
        if found:
            numberFou[real_v]+=1
    mylist=[]
    for i in numberTot:
        mylist.append((numberTot[i],numberFou[i]))
    mylist.sort(reverse=True)
    ratio=[r/e for e,r in mylist]
    for i,v in enumerate(ratio):
        average[i//group].append(v)
    ratio2=[]
    for key in average:
        ratio2.append(sum(average[key])/len(average[key]))
    pos=[(1/2+n)*group for n in range(len(ratio2))]
    ax.bar(range(len(mylist)),list(zip(*mylist))[0],linewidth=0,width=1)
    ax.bar(range(len(mylist)),list(zip(*mylist))[1],linewidth=0,color='r',width=1)
    axt=ax.twinx()
    axt.plot(range(len(mylist)),ratio,color='g')
    axt.plot(pos,ratio2,color='r')
    #second graph :
    ratio2.reverse()
    for i,v in enumerate(ratio2):
        if v!=0:
            until=int(pos[len(ratio2)-i-1])
            break
    ratio=[r/e for e,r in mylist][:until]
    average=collections.defaultdict(list)
    for i,v in enumerate(ratio):
        average[i//group].append(v)
    ratio2=[]
    for key in average:
        ratio2.append(sum(average[key])/len(average[key]))
    pos=[(1/2+n)*group for n in range(len(ratio2))]
    ax2.bar(range(until),list(zip(*mylist))[0][:until],linewidth=0,width=1)
    ax2.bar(range(until),list(zip(*mylist))[1][:until],linewidth=0,color='r',width=1)
    axt=ax2.twinx()
    axt.plot(range(until),ratio,color='g')
    average=collections.defaultdict(list)
    axt.plot(pos,ratio2,color='r')
        
        
    
    