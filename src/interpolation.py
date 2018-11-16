'''
Data Interpolation
'''


def interpolate(dataFrame, colsToInterpolate):
    '''
        Data Interpolation.
            Removing any NaN value in the data
        :param dataFrame: the data frame received from main function
        :param colsToInterpolate: the names of columns to be normalized
        :returns dataFrame: interpolated data frame
    '''
    for col in colsToInterpolate:
        dataFrame[col] = dataFrame[col].interpolate('spline',order=2)

    return dataFrame