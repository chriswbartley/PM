import numpy as np

def calc_mcc_interp(predXs,model,imonoFeat, uniqfeatvals,outlierdetection,calc_change_pt_extents=False) :
#calc_mcc_interp_pmsvm_rbf - Estimates the effect of a given feature on a
#   model's output class.
# DESCRIPTION:
#   This is an implementation of the summary monotonicity effects of a
#   feature at all datapoints built on the partial monotonicity framework
#   proposed in Bartley et al. 2016 'Effective Knowledge Integration in 
#   Support Vector Machines for Improved Accuracy' (submitted to
#   ECMLPKDD2016).In particular it enables the disagreement statistic
#   (Equation 10) to be calculated.
# 
#   For a given training data and PM-SVM model this classes each data point
#   as one of: (a) Not Changing in either direction (b) Monotone Increasing 
#   (in one or both directions) (c) Monotone Decreasing (in one or both 
#   directions) or (d) Monotone Increasing and Decreasing (one in each 
#   direction). When used with the unconstrained SVM model, this information 
#   can be used to calculate the disagreement statistic (Equation 10). The
#   information can also be used to summarise the impact of a feature on
#   the output class over the input space (weighted by the joint pdf).
#
#
# INPUTS:
#    predXs_copy - TxP matrix of test data on which to assess monotonicity.
#    model - an instance of an object with a method 'predict(Xs)'
#    imonoFeat - feature for which MCC is to be calculated
#    uniqfeatvals - An increasing vector of feature values that will be
#       used as the feature values where the def is tested. Use [] for
#       this to be deternp.mined automatically.
#    outlierdetection - 'od_on' will use outlier detection when deternp.mining 
#       the limits of each feature test (ie
#       global np.min and np.max will not necessarily be used, based on outlier 
#       detection usinga boxcox normalised 3 stdev definition for outlier.
#               
# OUTPUTS:
#   summary - 4x1 matrix giving the percentage (of the T test points) where
#       the def is: [NoChangeInEitherDirn , MonotoneIncreasing (one
#       or both directions),MonotoneDecreasing (one or both directions),
#       MonotoneIncrDec (one direction each)]. For an (increasing) feature,
#       this can be used to calculate the disagreement percentage (Equation
#       10 in paper): 
#       dis=(MonotoneDecreasing+0.5*MonotoneIncrDec)/(MonotoneIncreasing+MonotoneDecreasing+MonotoneIncrDec)
#   movements - Tx2 matrix showing directino of change in f(x) in
#       decreasing (column 1) and increasing (column 2) feature directions.
#       Values are -1 (first change is a decrease in f(x), 0 (no change), or +1
#       (first change is an increase in f(x).
#   change_pts - Tx4 matrix with locations of change and extent of region. Note the extent
#           values (cols 1 and 3) are only accurate if calc_change_pt_extents=True. 
#       col 0: negative direction change pt (hyperplane) -if any, otherwise -99. If 
#           movement value is 0 (no change) this will be -99.
#       col 1: negative direction extent - The point at which the class reverts back. If 
#           movement value is 0 (no change), this will be -99.
#       col 2: positive direction change pt (hyperplane) -if any, otherwise -99. If 
#           movement value is 0 (no change) this will be -99.
#       col 3: positive direction extent - The point at which the class reverts back. If 
#           movement value is 0 (no change), this will be -99.
#   pred_ys - a T sized array containing the predicted class for the test points
# Other m files needed:  outlier_limits, boxcox
# See also: nil
# Author: Chris Bartley
# University of Western Australia, School of Computer Science
# email address: christopher.bartley@research.uwa.edu.au
# Website: http://staffhome.ecm.uwa.edu.au/not 19514733/
# Last revision: 3-May-2016
    predXs_copy=predXs.copy() # ensure copy as 
    # calculate uniqfeatvals, if required
    if len(uniqfeatvals)==0: # need to calculate uniqfeatvals
        vals=np.sort(predXs_copy.copy()[:,imonoFeat-1],axis=0)
        uniqfeatvals=np.unique(vals)
        if len(uniqfeatvals)>10: # treat as continuous variable
            if  outlierdetection=='od_on':
                [lowlim,upplim]=outlier_limits(vals)
                origvals=vals
                vals=vals[vals>=lowlim]
                vals=vals[vals<=upplim]
                if len(vals)<0.7*len(origvals):
                     error('UNUSUAL NUMBER OF VALUES LOST IN OUTLIER REMOVAL - PLEASE CHECK!')
                minx=np.min(vals)
                maxx=np.max(vals)
            else: # no outlier detection, use full global extrema
               minx=np.min(uniqfeatvals)
               maxx=np.max(uniqfeatvals)
            resolu=30.0
            uniqfeatvals=np.arange(minx,maxx,(maxx-minx)/resolu)
    #if uniqfeatvals.shape[1]>uniqfeatvals.shape[0]:
    #    uniqfeatvals=uniqfeatvals.T
    # get global extents
    global_max=np.max(uniqfeatvals)
    global_min=np.min(uniqfeatvals)
    # for each datapoint, calculate m(x) monotonicity def (0,0.5,1)
    movements=np.zeros([predXs_copy.shape[0],2])
    change_pts=np.zeros([predXs_copy.shape[0],4])-99.
    pred_ys=np.zeros(predXs_copy.shape[0])-99.
    #nonmonopts=np.zeros(0,3)
    for i in np.arange(predXs_copy.shape[0]):
        # get the current value at x_i
        currx_i_p=np.double(predXs_copy[i,imonoFeat-1])
        # get all predictions for uniq vals
        # option one - feat variants
#         tic
#         predictedys_uniqvals=predict_consvm_rbf_featvars(predXs_copy(i,:),alphas,betas,MCs,b,y,X,kf,imonoFeat,uniqfeatvals)   
#         toc
        # option 2: faster prediction tecchnique (0.02 vs 0.15 sec)
#         tic
        predXvariants=np.tile(predXs_copy[i,:].copy(),(len(uniqfeatvals),1))
        predXvariants[:,imonoFeat-1]=uniqfeatvals
        predictedys_uniqvals=model.predict(predXvariants)#  predict_consvm_rbf(predXvariants,alphas,betas,MCs,b,y,X,kf)
        # get this prediction for curr X
        origy=model.predict(predXs_copy[i,:])#predict_consvm_rbf(predXs_copy(i,:),alphas,betas,MCs,b,y,X,kf)
        pred_ys[i]=origy
        # look at prior values then post values
        for searchdirn in [-1 ,+1]:
            found_change_pt=False
            found_change_extents=False
            # search in potential non-monotone direction first
            if searchdirn==1:
                nextxvals=np.sort(uniqfeatvals[uniqfeatvals>currx_i_p])
            else: # search backwards
                nextxvals=np.sort(uniqfeatvals[uniqfeatvals<currx_i_p])
                nextxvals=nextxvals[::-1] # reverse array
            if len(nextxvals)==0:
                k=0
            else: # have some prior values to test
                ks_all=len(nextxvals)
                lasty=origy
                for k in np.arange(ks_all ) :
                    x_k=predXs_copy[i,:].copy()
                    prior_feat_val=nextxvals[k]
                    x_k[imonoFeat-1]=prior_feat_val
                    ind=np.where(uniqfeatvals==prior_feat_val) #uniqfeatvals.index(prior_feat_val) #find(uniqfeatvals==prior_feat_val)
                    predy_prior=predictedys_uniqvals[ind] #(r,1)
                    if predy_prior != lasty:
                        if not found_change_pt: #this is the change pt
                            if searchdirn==-1: # looking prior
                                movements[i,0]=-1 if predy_prior<lasty else +1
                                change_pts[i,0]=prior_feat_val
                            else:
                                movements[i,1]= -1 if predy_prior<lasty else +1
                                change_pts[i,2]=prior_feat_val
                            if not calc_change_pt_extents: # we can stop, we know the direction of change
                                break
                            else: # keep going to find the extent of the region
                                found_change_pt=True
                                lasty=predy_prior
                        else: # this is the extent
                            found_change_extents=True
                            change_pts[i,1 if searchdirn==-1 else 3]=nextxvals[k-1] #prior_feat_val
                            break
                if calc_change_pt_extents and found_change_pt and not found_change_extents: # we didn't find the extents, use the global min max
                    change_pts[i,1 if searchdirn==-1 else 3]=global_min if searchdirn==-1 else global_max
        if currx_i_p== global_min and movements[i,0]!=0:
            print('problem!')
    # calculate summary
    numXs=predXs_copy.shape[0]
    summary=[0., 0., 0., 0.]
    summary[0]=sum(np.logical_and(movements[:,0]==0 , movements[:,1]==0 ))/np.double(numXs) # no change
    summary[1]=sum(np.logical_or(np.logical_or(np.logical_and(movements[:,0]<0 , movements[:,1]==0) , np.logical_and(movements[:,0]==0 , movements[:,1]>0)), np.logical_and(movements[:,0]<0 , movements[:,1]>0)))/np.double(numXs) # increasing
    summary[2]=sum(np.logical_or(np.logical_or(np.logical_and(movements[:,0]>0 , movements[:,1]==0) , np.logical_and(movements[:,0]==0 , movements[:,1]<0)), np.logical_and(movements[:,0]>0 , movements[:,1]<0)))/np.double(numXs) # decreasing
    summary[3]=sum(np.logical_or(np.logical_and(movements[:,0]>0 , movements[:,1]>0) , np.logical_and(movements[:,0]<0 , movements[:,1]<0)))/np.double(numXs) # incr-decreasing or dec-incr
    check=sum(summary)
    if (abs(check-1.))>1e-4:
        print ('***************** MCC INTERP CHECK IS WRONG *****************') 
             
    return [summary,movements,change_pts,pred_ys] 