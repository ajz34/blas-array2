import numpy as np
import oapackage


def taguchi(levels, run_size, nkeep=100, strength=2):
    number_of_factors = len(levels)
    factor_levels = levels
    strength = strength
    
    arrayclass = oapackage.arraydata_t(factor_levels, run_size, strength, number_of_factors)

    arraylist = [arrayclass.create_root()]

    # Extend arrays and filter based on D-efficiency
    options = oapackage.OAextend()
    options.setAlgorithmAuto(arrayclass)
    
    for extension_column in range(strength + 1, number_of_factors + 1):
        print("extend %d arrays with %d columns with a single column" % (len(arraylist), arraylist[0].n_columns))
        arraylist_extensions = oapackage.extend_arraylist(arraylist, arrayclass, options)
    
        # Select the best arrays based on the D-efficiency
        dd = np.array([a.Defficiency() for a in arraylist_extensions])
        ind = np.argsort(dd)[::-1][0:nkeep]
        selection = [arraylist_extensions[ii] for ii in ind]
        dd = dd[ind]
        print(
            "  generated %d arrays, selected %d arrays with D-efficiency %.4f to %.4f"
            % (len(arraylist_extensions), len(ind), dd.min(), dd.max()),
            flush=True
        )
    
        arraylist = selection
    return np.array(arraylist[0])


def taguchi_by_list(lst, run_size, nkeep=100, strength=2):
    lst_levels = [len(l) for l in lst]
    lst_taguchi = taguchi(lst_levels, run_size, nkeep, strength)
    return [[lst[n][i] for (n, i) in enumerate(l)] for l in lst_taguchi]
