import numpy as np

bias_lists = [["c2_b2_f", "c2_b1_b2", "c2_b1_b1",  
               "c2_b1_f", "c2_b1_f", "c1_b1_b1_f",
               "c1_b2_f", "c1_b1_b2", "c1_b1_b1",
               "c2_b1_b1_f", "c1_b1_f"], 
              ["c2_b1_f_f", "c1_f_f", "c1_f_f_f",
               "c2_f_f", "c2_f_f_f", "c1_f_f",
               "c1_b1_f_f"], 
              ["c1_c1_f_f", "c2_c2_b1_f", "c2_c1_b1_f",  
               "c2_c1_b1", "c2_c1_b2", "c2_c2_f_f",
               "c1_c1_f", "c2_c2_b1", "c2_c2_b2",
               "c2_c2_f", "c2_c1_b1_f", "c2_c1_f",
               "c1_c1_b1_f", "c1_c1_b1", "c1_c1_b2",
               "c1_c1_f_f"], 
              ["c1_c1_bG2", "c2_c2_bG2", "c2_c1_bG2"], 
              ["c1_b1_bG2", "c1_bG2_f", "c2_bG2_f", 
               "c2_b1_bG2"], 
              ["b1_f_f", "b1_b1_f_f", "b1_b1_b2", 
               "b2_f_f", "b1_b1_b1", "b1_b1_b1_f",
               "b1_b1_f", "b1_f_f_f", "f_f_f",
               "f_f_f_f", "b1_b2_f"], 
              ["bG2_f_f", "b1_b1_bG2", "b1_bG2_f"]]

def group_info(group, file_list=False):
    '''
    Args:
        group (int) : Group identifier.
        file_list (bool) : If ``True`` returns list of file containing
         the group kernels. Default is ``False``.

    Returns:
        Information about the kernel group.
    '''
    
    if file_list:
        return bias_lists[group]
    else:
        return len(bias_lists[group])

def combine_kernels(kernel_groups, f, b1=None, b2=None, bG2=None, c1=None, c2=None,
                    groups=None):
    '''
    Function for combing bias parameters and kernels.

    Args:
        kernel_groups (list) : List of arrays containing kernels.
         Each list element should have shape ``(n_ker, n_samp, n_k)``.
         With ``n_samp`` being the number of predictions to be made,
         ``n_ker`` being the number of kernels in that group,
         and ``n_k`` being the number of k-values.
        f (array) : Array containing growth rate.
         Should habe shape ``(n_samp)``.
        b1 (array) : Array containig b1 values.
         Should habe shape ``(n_samp)``. Default is ``None``.
        b2 (array) : Array containig b2 values.
         Should habe shape ``(n_samp)``. Default is ``None``.
        bG2 (array) : Array containig bG2 values.
         Should habe shape ``(n_samp)``. Default is ``None``.
        c1 (array) : Array containig c1 values.
         Should habe shape ``(n_samp)``. Default is ``None``.
        c2 (array) : Array containig c2 values.
         Should habe shape ``(n_samp)``. Default is ``None``.
        groups (list) : List of ``int`` that correspond to the group ids
         for the groups in ``kernel_groups``. Default is ``None``,
         in which case it is assumed all groups are passed.

    Returns:
        Array containig the combined kernels and bias parameters.
        Each list item will have shape ``(n_samp, n_k)``.
    '''

    # Store passed bias parameters in dict.
    bias_dict = {"b1":b1, "b2":b2, "bG2":bG2, "c1":c1, "c2":c2, "f":f}

    # If passsed groups is None. Assume all.
    if groups is None:
        groups = np.arange(len(bias_lists))

    result = 0 

    # Loop over the groups
    for list_id, group in enumerate(groups):

        # Determine what bias parameters are required for each kernel and
        # store in a list such that the bias_dict can be indexed.
        bias_list = [i.split("_") for i in group_info(group, file_list=True)]

        for component_id, components in enumerate(bias_list):
    
            bi = 1
            for bij in components:

                bi *= bias_dict[bij]
            result += bi.reshape(-1,1)*kernel_groups[list_id][component_id]

    return result

