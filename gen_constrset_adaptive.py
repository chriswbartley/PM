import numpy as np
from calc_mcc_interp import calc_mcc_interp
def gen_constrset_adaptive(Xcopy, numPairs,incrFeatures,decFeatures,model,params):
#gen_constrset_adaptive - Generates a set of non-conjunctive (univariate)
# constraints for use with constrained SVM, based on knowledge of the unconstrained SVM
# model.
# DESCRIPTION:
#   This generates a MxPx2 matrix which represents a set of M constraints to be 
#   used with the constrained SVM algorithm (train_consvm_rbf). Each constraint is
#   the pair of points (x_m,x_m') where x_m=MCs(m,:,1) and x_m'=MCs(m,:,2).
#   train_consvm_rbf() can then be used to train an SVM for which f(x_m')>=f(x_m) 
#   is guaranteed for this set of constraints.
#   This algorithm is used for the AD adaptivecontraints as described in Bartley 
#   et al. 2016 'Effective Knowledge Integration in Support Vector Machines 
#   for Improved Accuracy' (submitted to ECMLPKDD2016).
#
#   In brief, it only creates constraints at training datapoints known to
#   be non-monotone in the unconstrained SVM model. See paper for more
#   details.
#
# INPUTS:
#    X - NxP matrix of training data
#    numPairs - the number of constraints to be created (M)
#    incrFeatures - p_inc scalar indicating which features (p_inc subset of
#    (1..P)) are INCREASING monotone
#    decFeatures - p_dec scalar indicating which features (p_dec subset of
#    (1..P)) are DECREASING monotone
#    model - an instance of an object with a method 'predict(Xs)'
#    params - a dict with the following variables:
#       'creation_mode' - either 'to_max_constraints' or 'num_per_NMT_pt'
#       'constraints_per_NMT_pt' - number of constraints to be created per 
#           monotone point, or 0 if in 'to_max_constraints' mode
#       'max_constraints' - max constraints to be created.
#       'end_pt_selection' - 'random' for random, or 'equal' for midpoint of passive NMR
#           
# OUTPUTS:
#    MCs - MxPx2 matrix representing a set of M constraints to be 
#   used with the constraned SVM algorithm (train_consvm_rbf). Each constraint is
#   the pair of points (x_m,x_m') where x_m=MCs(m,:,1) and x_m'=MCs(m,:,2).
#
# EXAMPLE:
#   To create 500 constraints for increasing features [2 5] and
#   decreasing features [3 7], with recommended option settings, and then solve the PM-SVM model:
#   First solve unconstrained SVM:
#       [SVM_alphas,betas, y_pred,SVM_bias,H, minEig] = train_consvm_rbf(ytrain,Xtrain,bc,kf,Xtest, [])
#   Then create constraints:
#       MCs=gen_constrset_pmsvm_adaptive(Xtrain, 500,[2 5],[3 7],{'randseln','NMTpt_prop','randseln','baseanchor_none','endanchor_none','base_noprior','od_off','ub_on','dupchk_on',kf,ytrain,Xtrain,SVM_alphas,SVM_bias}) 
#   Then solve constrained SVM with constraints:
#       [alphas,betas, y_pred,bias,H, minEig] = train_consvm_rbf(ytrain,Xtrain,bc,kf,Xtest, MCs)
#
# Other m-files required:  iif
#
# See also: train_consvm_rbf
# Author: Chris Bartley
# University of Western Australia, School of Computer Science
# email address: christopher.bartley@research.uwa.edu.au
# Website: http://staffhome.ecm.uwa.edu.au/19514733/
# Last revision: 15 May 2016
#------------- BEGIN CODE --------------
    n=len(Xcopy)
    constrFeats=list(incrFeatures)+list(decFeatures)
    
    # 10: GET THE NON-MONOTONE POINTS
    X_nmt_indexes=dict()
    nmt_exents_lbound=dict()    
    nmt_exents_ubound=dict()  
    uniq_mono_feat_vals=dict()
    ttl_nmt_pts=0
    for imonoFeat in constrFeats:
        [summary,movements,change_pts,pred_ys] =calc_mcc_interp(predXs=Xcopy,model=model,imonoFeat=imonoFeat, uniqfeatvals=[],outlierdetection='od_off',calc_change_pt_extents=True)
        increasing=imonoFeat in list(incrFeatures)
        nmt_indexes=list()
        nmt_lbound=list()
        nmt_ubound=list()
        for dirn in [-1,1]:
            col=0 if dirn==-1 else 1
            potentially_nmt_in_this_dirn_indexes=pred_ys==dirn if increasing else pred_ys==-dirn
            potentially_nmt_indexes=np.arange(n)[potentially_nmt_in_this_dirn_indexes]
            actually_nmt_indexes=potentially_nmt_indexes[movements[potentially_nmt_in_this_dirn_indexes,col]==(-dirn  if increasing else dirn)]
            nmt_indexes.extend(actually_nmt_indexes)
            lbounds=[min(pair) for pair in zip(change_pts[actually_nmt_indexes,1 if dirn==-1 else 2],change_pts[actually_nmt_indexes,0 if dirn==-1 else 3])]   
            ubounds=[max(pair) for pair in zip(change_pts[actually_nmt_indexes,1 if dirn==-1 else 2],change_pts[actually_nmt_indexes,0 if dirn==-1 else 3])]               
            nmt_lbound.extend(lbounds )
            nmt_ubound.extend(ubounds) #[max(pair) for pair in zip(change_pts[actually_nmt_indexes,1 if dirn==-1 else 2],change_pts[actually_nmt_indexes,0 if dirn==-1 else 3])]    )
            if np.sum(Xcopy[actually_nmt_indexes,imonoFeat-1]==lbounds)>0:
                print('err!!')
        X_nmt_indexes[imonoFeat]=np.asarray(nmt_indexes,dtype='int')
        nmt_exents_lbound[imonoFeat]=np.asarray(nmt_lbound)
        nmt_exents_ubound[imonoFeat]=np.asarray(nmt_ubound)
        ttl_nmt_pts=ttl_nmt_pts+len(nmt_indexes)
        uniq_mono_feat_vals[imonoFeat]=np.unique(Xcopy[:,imonoFeat-1])
    
    # 20: CALCULATE THE NUMBER OF CONSTRAINTS PER NMT POINT
    if params['creation_mode']=='num_per_NMT_pt':
        num_constr_per_pt=params['constraints_per_NMT_pt']
        num_constr=num_constr_per_pt*ttl_nmt_pts
    elif params['creation_mode']=='to_max_constraints':
        num_constr_per_pt=np.ceil( np.double(params['max_constraints'])/np.double(ttl_nmt_pts))
        num_constr=num_constr_per_pt*ttl_nmt_pts
    else:
        raise ValueError('Creation mode ' + params['creation_mode'] + ' not supported at the moment.') 
    
    # 30: BUILD CONSTRAINTS
    MCs=np.zeros([num_constr,Xcopy.shape[1],2])
    irow=0
    for pt in np.arange(num_constr_per_pt):
        for imonoFeat in constrFeats:
            if len(X_nmt_indexes[imonoFeat])>0:
                increasing=imonoFeat in list(incrFeatures)
                base_pts=Xcopy[X_nmt_indexes[imonoFeat],:]
                uniq_vals=uniq_mono_feat_vals[imonoFeat]
                lbounds=nmt_exents_lbound[imonoFeat]
                ubounds=nmt_exents_ubound[imonoFeat]
                MCs[irow:irow+len(X_nmt_indexes[imonoFeat]),:,0]=base_pts
                MCs[irow:irow+len(X_nmt_indexes[imonoFeat]),:,1]=base_pts
                constrfeatvals=np.zeros([len(base_pts),2])
                constrfeatvals[:,0]=base_pts[:,imonoFeat-1]
                if params['end_pt_selection']=='random':
                    endpts=lbounds+0.15*(ubounds-lbounds)+[ai*bi for ai,bi in zip(np.random.rand(len(base_pts)), 0.7*(ubounds-lbounds))]
                elif params['end_pt_selection']=='equal':
                    endpts=lbounds+0.5*(ubounds-lbounds)
                option=1
                if option ==1:
                    # OPTION 1: just tret as real numbers
                    constrfeatvals[:,1]=endpts
                else:
                    #OPTION 2: SNAPS TO THE CLOSEST UNIQUE VALUE:
                    deviations_from_std=np.abs(np.tile(endpts.reshape([len(endpts),1]),(1,len(uniq_vals)))-np.tile(uniq_vals.reshape([1,len(uniq_vals)]),(len(endpts),1)))
                    std_indxs_endpts=np.argmin(deviations_from_std,1)
                    deviations_of_basepts=np.abs(np.tile(constrfeatvals[:,0].reshape([len(constrfeatvals[:,0]),1]),(1,len(uniq_vals)))-np.tile(uniq_vals.reshape([1,len(uniq_vals)]),(len(constrfeatvals[:,0]),1)))
                    std_indxs_of_base=np.argmin(deviations_of_basepts,1)
                    std_indxs_endpts[std_indxs_endpts==std_indxs_of_base]=[std_indxs_endpts[i]-1 if base_pts[i,imonoFeat-1]>endpts[i] else std_indxs_endpts[i]+1 for i in np.nonzero(std_indxs_endpts==std_indxs_of_base)]
                    constrfeatvals[:,1]=[uniq_vals[i] for i in std_indxs_endpts ]
                
                MCs[irow:irow+len(X_nmt_indexes[imonoFeat]),imonoFeat-1,0]=np.min(constrfeatvals,axis=1) if increasing else np.max(constrfeatvals,axis=1)
                MCs[irow:irow+len(X_nmt_indexes[imonoFeat]),imonoFeat-1,1]=np.max(constrfeatvals,axis=1) if increasing else np.min(constrfeatvals,axis=1)
                irow=irow+len(base_pts)
    # 40: TRIM TO MAXIMUM (if required)
    if params['creation_mode']=='to_max_constraints':
        if num_constr>params['max_constraints']:
            MCs=MCs[0:params['max_constraints'],:,:]
    # 50: RETURN RESULT
    return MCs
