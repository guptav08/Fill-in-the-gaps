#!/usr/bin/env python
# coding: utf-8

# In[1]:


from cmath import nan
from xml.etree.ElementTree import tostring
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import euclidean
from scipy.special import binom
#from fastdtw import fastdtw
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.impute import KNNImputer
from scipy.interpolate import PchipInterpolator, interp1d, BSpline
from scipy.stats import pearsonr
from scipy.signal import argrelextrema
from scipy.stats import pearsonr
from scipy.signal import find_peaks
import random
import glob

def find_continuous_blocks(data, threshold=1):
    """
    Finds all continuous data blocks where the time difference between records is less than or equal to 'threshold' seconds.
    It returns the blocks along with their lengths in seconds.
    
    :param data: DataFrame containing 'time' and 'test' columns
    :param threshold: Maximum allowed gap in seconds between consecutive records
    :return: List of continuous blocks with their length (duration)
    """
    # Sort data by 'time' to ensure proper order

    data.sort_values(by='datetime', inplace=True)
    
    continuous_blocks = []
    current_block = [data.iloc[0]]  # Start with the first row
    print("sorted")
    for i in range(1, len(data)):
        time_diff = data['datetime'].iloc[i] - data['datetime'].iloc[i-1]
        
        if time_diff <= threshold:
            # If the time difference is within the threshold, add to current block
            current_block.append(data.iloc[i])
        else:
            # If there's a gap larger than threshold, finalize the current block
            if len(current_block) > 1:
                start_time = current_block[0]['datetime']
                end_time = current_block[-1]['datetime']
                block_length = end_time - start_time
                continuous_blocks.append((start_time, end_time, block_length))
                print("New block: " +  str(start_time) + ", " + str(end_time) + ", " + str(block_length))
            # Start a new block with the current entry
            current_block = [data.iloc[i]]
    
    # Check the last block after finishing the loop
    if len(current_block) > 1:
        start_time = current_block[0]['datetime']
        end_time = current_block[-1]['datetime']
        block_length = end_time - start_time
        continuous_blocks.append((start_time, end_time, block_length))
    
    return continuous_blocks

def pick_numbers(min_val, max_val, num_picks, min_gap):
    # Initialize an empty list to hold the selected numbers
    selected_numbers = []
    
    while len(selected_numbers) < num_picks:
        # Pick a random number within the range
        num = random.randint(min_val, max_val)
        print("Checking num: " + str(num))
        # Check if the number is at least `min_gap` away from all previously selected numbers
        if all(abs(num - selected) >= min_gap for selected in selected_numbers):
            selected_numbers.append(num)
            print("Added num: " + str(num))
    
    return selected_numbers

# Helper function to find peaks and troughs in control points
def find_extrema(control_points):
    peaks = argrelextrema(np.array(control_points), np.greater)[0]
    troughs = argrelextrema(np.array(control_points), np.less)[0]
    return peaks, troughs


# Weighted Bézier curve function with control points
def bezier_curve_with_extrema(control_points, weights, segment_length=30):
    n = len(control_points) - 1  # Degree of the Bézier curve
    t_values = np.linspace(0, 1, segment_length)
    bezier_values = np.zeros(segment_length)
    
    print("-----------bez")
    print(t_values)

    for i in range(n + 1):
        binomial_coeff = binom(n, i)

        bezier_values += (weights[i] * control_points[i] *
                          binomial_coeff * (t_values ** i) * ((1 - t_values) ** (n - i)))
        
    # Normalize by weights to complete rational Bézier computation
    weight_sum = sum(weights[i] * binom(n, i) * (t_values ** i) * ((1 - t_values) ** (n - i))
                     for i in range(n + 1))
    bezier_values /= weight_sum
    print(bezier_values)
    return bezier_values

# In[2]:
file_name = "/result.txt"
gapTime = 1500
MaxGapTime = 3600


"""
#for average results
totalRMSE = 0
totalEDM = 0
totalPeak = 0
totalMetrik = 0
totalClark15 = 0
totalClark30 = 0
totalClark50 = 0
totalClarkMore = 0


totalRMSELinear = 0
totalEDMLinear = 0
totalPeakLinear = 0
totalMetrikLinear = 0
totalClark15Linear = 0
totalClark30Linear = 0
totalClark50Linear = 0
totalClarkMoreLinear = 0"""


totalAmount = 0
resultsPCHIP = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
resultBezier = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
resultLinear = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
resultKNN = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
resultBSpline = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
resultBasicCHIP = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
resultWords = ["RMSE: ", "EDM: ", "Peak: ", "Metric: ", "Clark15: ", "Clark30: ", "Clark50: ", "Clark Over 50: "]

data_path_1 = 'path to the data'
data_path_2 = 'path to the data'

fileList = glob.glob(data_path_2 + "/*.csv")

print(fileList)

for i in range(0, len(fileList)):

    data = pd.read_csv(fileList[i], sep=",")
    data = data.drop(data[data['HR'] == 0].index)
    data = data.drop(data[data['HR'] == 0.0].index)
    data = data.dropna(subset=['HR'])
    data = data.drop(data[data['HR'] == nan].index)
    data = data.reset_index(drop=True)
    data.sort_values(by='datetime', inplace=True)
    data.to_csv(fileList[i], sep=",")
    #data = pd.read_csv(data_path_1)


    #find blocks of at least x samples without gaps
    blocks = find_continuous_blocks(data)
    #take a random data point in that block

    # In[3]:
    

    print("Begin of patient ")
    print(blocks)
    print("AFTER BLOCKS")
    # Print out the continuous blocks and their lengths
    for start, end, length in blocks:
        #print(f"Block from {start} to {end}, Length: {length} seconds")
        
        if(length > 4*gapTime):
            print("len: " + str(length))
            amount = (length-gapTime) / (12*MaxGapTime)
            amount = (length) / (4*MaxGapTime)
            print("len: " + str(length))
            print("amount: " + str(amount))
            numbers = pick_numbers(start+gapTime,end-(2*gapTime),amount, 3* gapTime)

            for j in range(0, len(numbers)):
                print(numbers[j])
             

                #number[1] is the starting point of the artificial gap
                #
                gapstart = data.loc[data['datetime'] == numbers[j]].index[0]
                print(gapstart)
                impute_range = range(gapstart,gapstart+gapTime) 
                control_range_before = range(gapstart - 1 - int((gapTime/2)),gapstart-1)
                control_range_after = range(gapstart+gapTime+1,gapstart+gapTime+1+int((gapTime/2)))


                hr_original = data['HR'].copy()
                hr_imputed = data['HR'].copy()
                #impute_range = range(28281,28341) 
                #control_range_before = range(28250,28280)
                #control_range_after = range(28342,28372)


                #-------------------------------------
                # Check if there is a data problem (only one value for a longer time)
                #---------------------------------------
                control_points_before = hr_imputed[control_range_before].values
                control_points_after = hr_imputed[control_range_after].values
                # Identify peaks and valleys in each control region separately
                peaks_before = argrelextrema(control_points_before, np.greater)[0]
                valleys_before = argrelextrema(control_points_before, np.less)[0]
                peaks_after = argrelextrema(control_points_after, np.greater)[0]
                valleys_after = argrelextrema(control_points_after, np.less)[0]

                if  peaks_before.size == 0 or valleys_before.size == 0 or peaks_after.size == 0 or valleys_after.size == 0:
                    print("No peaks or Valleys")
                    continue

                #------------------------------------
                # Linear Interpolation for Comparison
                #------------------------------------
                linear_interpolator = interp1d(
                    [impute_range.start, impute_range.stop - 1],  # Now using 931 to 946 as control points
                    [hr_imputed[impute_range.start], hr_imputed[impute_range.stop - 1]],  # Values at 931 and 946
                    kind='linear',
                    fill_value="extrapolate"
                )
                # Perform linear interpolation over the imputation range
                linear_imputed_values = linear_interpolator(np.arange(impute_range.start, impute_range.stop))

                # Clip linear imputed values 40 to 160 bpm
                linear_imputed_values_clipped = np.clip(linear_imputed_values, 40, 160)
                hr_linear_imputed = hr_imputed.copy()
                hr_linear_imputed[impute_range] = linear_imputed_values_clipped


                #---------------------------------
                # PCHIP
                #---------------------------------

                # Separate peaks and valleys from control range before and after the imputation region
                control_points_before = hr_imputed[control_range_before].values
                control_points_after = hr_imputed[control_range_after].values



                # Identify peaks and valleys in each control region separately
                peaks_before = argrelextrema(control_points_before, np.greater)[0]
                valleys_before = argrelextrema(control_points_before, np.less)[0]
                extrema_indices_before = np.sort(np.concatenate((peaks_before, valleys_before)))
                extrema_points_before = control_points_before[extrema_indices_before]



                peaks_after = argrelextrema(control_points_after, np.greater)[0]
                valleys_after = argrelextrema(control_points_after, np.less)[0]
                extrema_indices_after = np.sort(np.concatenate((peaks_after, valleys_after)))
                extrema_points_after = control_points_after[extrema_indices_after]


                # Reverse the points before and after separately
                extrema_points_before_flipped = extrema_points_before[::-1]
                extrema_points_after_flipped = extrema_points_after[::-1]

                # Combine the reversed points
                extrema_points_combined = np.concatenate([extrema_points_before_flipped, extrema_points_after_flipped])

                # Map extrema points to the imputation range in flipped order
                extrema_indices_in_impute = np.linspace(impute_range.start, impute_range.stop - 1, len(extrema_points_combined), dtype=int)
                hr_imputed[extrema_indices_in_impute] = extrema_points_combined

                # Linear interpolation for the first and last 5 points in the imputation range
                linear_start_indices = np.arange(impute_range.start, impute_range.start + 5)
                linear_end_indices = np.arange(impute_range.stop - 5, impute_range.stop)

                linear_start_values = np.linspace(hr_imputed[impute_range.start - 1], extrema_points_combined[0], 5)
                linear_end_values = np.linspace(extrema_points_combined[-1], hr_imputed[impute_range.stop], 5)

                # Middle section using PCHIP interpolation
                # Define the middle range excluding the first and last 5 values
                pchip_indices = np.arange(impute_range.start + 5, impute_range.stop - 5)
                x_fixed = np.concatenate(([impute_range.start - 1], extrema_indices_in_impute, [impute_range.stop]))
                y_fixed = np.concatenate(([hr_imputed[impute_range.start - 1]], extrema_points_combined, [hr_imputed[impute_range.stop]]))

                # Perform PCHIP interpolation over the middle section
                pchip_interpolator = PchipInterpolator(x_fixed, y_fixed)
                pchip_values = pchip_interpolator(pchip_indices)

                # Combine all imputed values: linear start, PCHIP middle, and linear end
                imputed_values = np.concatenate([linear_start_values, pchip_values, linear_end_values])

                # Clip imputed values to a reasonable HR range
                imputed_values_clipped = np.clip(imputed_values, 40, 160)
                hr_imputed[impute_range] = imputed_values_clipped



                #------------------------------
                # Rational Bezier
                #------------------------------
                # imputation range and control points range
                start_index = impute_range.start
                end_index = impute_range.stop
                control_start = int(impute_range.start - 1 - (gapTime/2))
                control_end = impute_range.start-1
                control_post_start =  impute_range.stop + 1
                control_post_end = int(impute_range.stop + 1 + (gapTime/2))
                hr_imputed_bezier = data['HR'].copy()


                # Collect control points from before and after the missing segment
                control_points_before = hr_original[control_start:control_end].values
                control_points_after = hr_original[control_post_start:control_post_end].values
                control_points = np.concatenate([control_points_before, control_points_after])

                # Find peaks and troughs in the control points for weighted emphasis
                peaks, troughs = find_extrema(control_points)


                # In[9]:


                # Initialize weights with emphasis on peaks and troughs
                weights = np.ones(len(control_points))
                weights[peaks] *= 1.5  
                weights[troughs] *= 1.5 

                # Define indices for linear and Bézier imputation sections
                linear_segment_length = 5
                middle_start = start_index + linear_segment_length
                middle_end = end_index - linear_segment_length

                # Linear interpolation for the first 5 values 
                linear_start_values = np.linspace(hr_original[control_end - 1], hr_original[control_post_start], num=linear_segment_length)

                # Rational Bézier curve for the middle section 
                segment_length = middle_end - middle_start
                imputed_values_middle = bezier_curve_with_extrema(control_points, weights, segment_length)

                # Linear interpolation for the last 5 values 
                linear_end_values = np.linspace(hr_original[control_end], hr_original[control_post_start], num=linear_segment_length)

                # Combine the imputed values
                imputed_values = np.concatenate([
                    linear_start_values,
                    imputed_values_middle,
                    linear_end_values
                ])

                # Insert the imputed values into the HR data
                hr_imputed_bezier[start_index:end_index] = imputed_values


                """
                # Linear interpolation for the first and last 5 points in the imputation range
                linear_start_indices = np.arange(impute_range.start, impute_range.start + 5)
                linear_end_indices = np.arange(impute_range.stop - 5, impute_range.stop)


                linear_start_values = np.linspace(hr_imputed_bezier[impute_range.start - 1], extrema_points_combined[0], 5)
                linear_end_values = np.linspace(extrema_points_combined[-1], hr_imputed_bezier[impute_range.stop], 5)

                # Middle section using PCHIP interpolation
                # Define the middle range excluding the first and last 5 values
                pchip_indices = np.arange(impute_range.start + 5, impute_range.stop - 5)
                x_fixed = np.concatenate(([impute_range.start - 1], extrema_indices_in_impute, [impute_range.stop]))
                y_fixed = np.concatenate(([hr_imputed_bezier[impute_range.start - 1]], extrema_points_combined, [hr_imputed_bezier[impute_range.stop]]))

                # Perform PCHIP interpolation over the middle section
                pchip_interpolator = PchipInterpolator(x_fixed, y_fixed)
                pchip_values = pchip_interpolator(pchip_indices)

                # Combine all imputed values: linear start, PCHIP middle, and linear end
                imputed_values = np.concatenate([linear_start_values, pchip_values, linear_end_values])

                # Clip imputed values to a reasonable HR range
                imputed_values_clipped = np.clip(imputed_values, 40, 160)
                hr_imputed_bezier[impute_range] = imputed_values_clipped"""

                #--------------------
                #Knn for comparison
                #--------------------
                hr_knn = data['HR'].copy()
                hr_knn.loc[impute_range] = np.nan 
                def dynamic_knn_imputation(column, impute_range, n_neighbors=5):
                    column = column.copy()  # Work on a copy
                    known_indices = column.dropna().index  # Indices of known values
                    for i in impute_range:
                        # Find the n_neighbors closest known indices to the current missing value
                        distances = abs(pd.Series(known_indices) - i)  # Compute distances to known indices
                        nearest_indices = distances.nsmallest(n_neighbors).index  # Get closest indices
        
                        # Compute the mean of the nearest neighbors
                        nearest_values = column.iloc[nearest_indices]
                        imputed_value = nearest_values.mean()
        
                        # Assign the imputed value
                        column.loc[i] = imputed_value

                    return column

                # Perform dynamic KNN imputation
                hr_knn_imputed = dynamic_knn_imputation(hr_knn, impute_range, n_neighbors=5)
                # Clip the imputed HR values to a realistic range (e.g., 40 to 160 bpm)
                hr_knn_imputed_clipped = np.clip(hr_knn_imputed, 40, 160)

                #-----------------------------------------------------------
                #B Splie
                #-----------------------------------------------------------

                known_indices = data['HR'].dropna().index  # Indices of known HR values

                #Remove indices from the impute_range to avoid them being used in interpolation (they will be imputed)
                known_indices = known_indices[~known_indices.isin(impute_range)]

                #Extract known HR values for interpolation
                x_known = known_indices
                y_known = data.loc[x_known, 'HR']

                degree = 3  
                spl = BSpline(x_known, y_known, degree)

                #Impute missing values in the range 28281 to 28341 using the B-Spline
                imputed_values_b_spline = spl(np.array(impute_range))

                #Clip pchip imputed values 40 to 160 bpm
                b_spline_imputed_values_clipped = np.clip(imputed_values_b_spline, 40, 160)
                hr_b_spline_imputed = data['HR'].copy()
                hr_b_spline_imputed[impute_range] = b_spline_imputed_values_clipped

                #---------------------------------------------------------
                # Basic Chip
                #---------------------------------------------------------
                
                known_indices = data['HR'].dropna().index  # Indices of known HR values

                #Remove indices from the impute_range to avoid them being used in interpolation (they will be imputed)
                known_indices = known_indices[~known_indices.isin(impute_range)]

                #Extract known HR values for interpolation
                x_known = known_indices
                y_known = data.loc[x_known, 'HR']

                #Create the PCHIP interpolator
                pchip_interpolator = PchipInterpolator(x_known, y_known)

                #Impute missing values in the range 28281 to 28341
                imputed_values__pchip = pchip_interpolator(np.array(impute_range))

                # Clip pchip imputed values 40 to 160 bpm
                pchip_imputed_values_clipped = np.clip(imputed_values__pchip, 40, 160)
                hr_pchip_imputed = data['HR'].copy()
                hr_pchip_imputed[impute_range] = pchip_imputed_values_clipped

                #--------------------------------------------------------
                #Plot
                #----------------------------------------------------------

                print("----plot-----")
                print(hr_original)
                print(hr_original[impute_range.start:impute_range.stop])

                plt.figure(figsize=(12, 6))

                # Plot the original HR values 28281:28341
                #plt.plot(hr_original.index[2000:3000], hr_original[2000:3000], label='Original Values', color='blue')
                plt.plot(hr_original.index[impute_range.start:impute_range.stop], hr_original[impute_range.start:impute_range.stop], label='Original Values', color='blue')
                #plt.plot(data['HR'][impute_range.start:impute_range.stop], label='Original Values', color='blue')

                # Plot the imputed HR values using PCHIP with peak/valley preservation
                plt.plot(hr_imputed.index[impute_range.start:impute_range.stop], hr_imputed[impute_range.start:impute_range.stop], label='Imputed Values with Peak/Valley Preservation (PCHIP)', color='red', linestyle='--')
                # Plot the linearly imputed HR values for comparison
                plt.plot(hr_linear_imputed.index[impute_range.start:impute_range.stop], hr_linear_imputed[impute_range.start:impute_range.stop], label='Linear Imputation', color='green', linestyle='--')
                # Plot the Bezier imputed HR values for comparison
                plt.plot(hr_imputed_bezier.index[impute_range.start:impute_range.stop], hr_imputed_bezier[impute_range.start:impute_range.stop], label='Bezier Imputation', color='Orange', linestyle='--')
                
                # Mark the mapped peaks and valleys from PCHIP (using flipped-mapping extrema)
                plt.scatter(extrema_indices_in_impute, extrema_points_combined, color='purple', label="Mapped Peaks/Valleys", zorder=5)
              
                # Titles and labels
                plt.title("Comparison of Original and Imputed Values")
                plt.xlabel("Index")
                plt.ylabel("HR Value")
                plt.legend()

                figureName = "fig" + str(i) + "at" + str(numbers[j]) + ".png"
                plt.savefig("C:/Users/grens/Spaces/Heart Rate Simulation Project/python scripts/Graphs/"+figureName)  # Save as a PNG file
              
                plt.close()
                


                # In[8]:

                """
                mse_pchip = mean_absolute_error(hr_original[impute_range], hr_imputed[impute_range])
                rmse_pchip = mean_squared_error(hr_original[impute_range], hr_imputed[impute_range],squared=False)
                print("Mean absolute Error for PCHIP Imputation:", mse_pchip)
                print("Mean Squared Error for pchip Imputation:", rmse_pchip)
                """

                # In[9]:


                # Function to calculate EDM for a segment of data
                def calculate_extremum_density(segment):
                    segment = np.array(segment)
                    max_count = np.sum((segment[1:-1] > segment[:-2]) & (segment[1:-1] > segment[2:]))
                    min_count = np.sum((segment[1:-1] < segment[:-2]) & (segment[1:-1] < segment[2:]))
                    return (max_count + min_count) / len(segment)

                # Updated function to calculate scores with peak alignment, RMSE, and EDM
                def calculate_custom_score_with_peak_alignment(original_values, imputed_values):
                    # RMSE
                    rmse_score = np.sqrt(mean_squared_error(original_values, imputed_values))
    
                    #EDM (using extremum density difference between original and imputed)
                    edm_score_original = calculate_extremum_density(original_values)
                    edm_score_imputed = calculate_extremum_density(imputed_values)
                    edm_score = abs(edm_score_original - edm_score_imputed)
    
                    #Peak Alignment Score with inf handling
                    def calculate_peak_alignment_score(orig, imp):
                        original_peaks, _ = find_peaks(orig)
                        imputed_peaks, _ = find_peaks(imp)
                        min_len = min(len(original_peaks), len(imputed_peaks))
                        if min_len > 0:
                            peak_diff = np.abs(orig[original_peaks[:min_len]] - imp[imputed_peaks[:min_len]])
                            peak_alignment_score = np.mean(peak_diff)
                        else:
                            peak_alignment_score = 15  # Set to 15 if no peaks match, instead of inf
                        return peak_alignment_score
    
                    peak_alignment_score = calculate_peak_alignment_score(original_values, imputed_values)
    
                    #Combined Score (Weighted Average of RMSE, EDM, and Peak Alignment Score)
                    weight_rmse = 1/3
                    weight_edm = 1/3
                    weight_peak = 1/3

                    combined_score = (weight_rmse * rmse_score +
                                      weight_edm * edm_score +
                                      weight_peak * peak_alignment_score)
    
                    return {
                        "RMSE Score": rmse_score,
                        "EDM Score": edm_score,
                        "Peak Alignment Score": peak_alignment_score,
                        "Combined Score": combined_score
                    }

                original_impute_values = hr_original.iloc[impute_range].values
                pchip_impute_values = hr_imputed.iloc[impute_range].values
                linear_impute_values = hr_linear_imputed.iloc[impute_range].values
                knn_impute_values = hr_knn_imputed_clipped.iloc[impute_range].values
                bspline_impute_values = hr_b_spline_imputed.iloc[impute_range].values
                chip_impute_values = hr_pchip_imputed.iloc[impute_range].values
                bezier_impute_values = hr_imputed_bezier.iloc[impute_range].values
                print(bezier_impute_values)
                total_nans = np.isnan(bezier_impute_values).sum()
                print(total_nans)


                # Calculate scores for both PCHIP and Linear Imputation methods
                print("testa")
                score_pchip = calculate_custom_score_with_peak_alignment(original_impute_values, pchip_impute_values)
                print("testa")
                score_linear = calculate_custom_score_with_peak_alignment(original_impute_values, linear_impute_values)
                print("testa")
                score_bezier = calculate_custom_score_with_peak_alignment(original_impute_values, bezier_impute_values)
                print("testa")
                score_bspline = calculate_custom_score_with_peak_alignment(original_impute_values, bspline_impute_values)
                print("testa")
                score_knn = calculate_custom_score_with_peak_alignment(original_impute_values, knn_impute_values)
                print("testa")
                score_chip = calculate_custom_score_with_peak_alignment(original_impute_values, chip_impute_values)

                """
                print("Scores for PCHIP Imputation:", score_pchip)
                print("Scores for Linear Imputation:", score_linear)

                # Compare Combined Scores
                if score_pchip["Combined Score"] < score_linear["Combined Score"]:
                    print("PCHIP interpolation is performing better.")
                else:
                    print("Linear interpolation is performing better.")
                """



                # Clarke Error Grid Analysis function with 10%, 20%, 30%, and >30% zones
                def clarke_error_grid_analysis(reference, predicted):
                    zone_15, zone_30, zone_50, zone_over_50 = 0, 0, 0, 0
    
                    for ref, pred in zip(reference, predicted):
                        if ref == 0:
                            continue
                        percent_diff = abs((pred - ref) / ref * 100)
        
                        # Define the zones based on percentage difference thresholds
                        if percent_diff <= 15:
                            zone_15 += 1
                        elif percent_diff <= 30:
                            zone_30 += 1
                        elif percent_diff <= 50:
                            zone_50 += 1
                        else:
                            zone_over_50 += 1

                    total_points = len(reference)
                    return {
                        "Zone ≤15% (Clinically Accurate)": zone_15 / total_points * 100,
                        "Zone ≤30% (Likely Acceptable)": zone_30 / total_points * 100,
                        "Zone ≤50% (Clinically Safe)": zone_50 / total_points * 100,
                        "Zone >50% (Clinically Significant Error)": zone_over_50 / total_points * 100
                    }

                # Apply CEGA on both Bézier and Linear Imputed values
                cega_pchip = clarke_error_grid_analysis(original_impute_values, pchip_impute_values)
                cega_bezier = clarke_error_grid_analysis(original_impute_values, bezier_impute_values)
                cega_linear = clarke_error_grid_analysis(original_impute_values, linear_impute_values)
                cega_bspline = clarke_error_grid_analysis(original_impute_values, bspline_impute_values)
                cega_knn = clarke_error_grid_analysis(original_impute_values, knn_impute_values)
                cega_chip = clarke_error_grid_analysis(original_impute_values, chip_impute_values)
                
                """
                # Output Clarke Error Grid Analysis results
                print("Clarke Error Grid Analysis for Pchip Imputed Values:")
                for key, value in cega_bezier.items():
                    print(f"{key}: {value:.2f}%")

                print("\nClarke Error Grid Analysis for Linear Imputed Values:")
                for key, value in cega_linear.items():
                    print(f"{key}: {value:.2f}%")
                """

                

                """
                totalRMSE += score_pchip["RMSE Score"]
                totalEDM += score_pchip["EDM Score"]
                totalPeak += score_pchip["Peak Alignment Score"]
                totalMetrik += score_pchip["Combined Score"]
                totalClark15 += cega_bezier["Zone ≤15% (Clinically Accurate)"]
                totalClark30 += cega_bezier["Zone ≤30% (Likely Acceptable)"]
                totalClark50 += cega_bezier["Zone ≤50% (Clinically Safe)"]
                totalClarkMore += cega_bezier["Zone >50% (Clinically Significant Error)"]
                totalAmount += 1

                totalRMSELinear += score_linear["RMSE Score"]
                totalEDMLinear += score_linear["EDM Score"]
                totalPeakLinear += score_linear["Peak Alignment Score"]
                totalMetrikLinear += score_linear["Combined Score"]
                totalClark15Linear += cega_linear["Zone ≤15% (Clinically Accurate)"]
                totalClark30Linear += cega_linear["Zone ≤30% (Likely Acceptable)"]
                totalClark50Linear += cega_linear["Zone ≤50% (Clinically Safe)"]
                totalClarkMoreLinear += cega_linear["Zone >50% (Clinically Significant Error)"]
                """

                resultsPCHIP[0] += score_pchip["RMSE Score"]
                resultsPCHIP[1] += score_pchip["EDM Score"]
                resultsPCHIP[2] += score_pchip["Peak Alignment Score"]
                resultsPCHIP[3] += score_pchip["Combined Score"]
                resultsPCHIP[4] += cega_pchip["Zone ≤15% (Clinically Accurate)"]
                resultsPCHIP[5] += cega_pchip["Zone ≤30% (Likely Acceptable)"]
                resultsPCHIP[6] += cega_pchip["Zone ≤50% (Clinically Safe)"]
                resultsPCHIP[7] += cega_pchip["Zone >50% (Clinically Significant Error)"]
                
                resultBezier[0] += score_bezier["RMSE Score"]
                resultBezier[1] += score_bezier["EDM Score"]
                resultBezier[2] += score_bezier["Peak Alignment Score"]
                resultBezier[3] += score_bezier["Combined Score"]
                resultBezier[4] += cega_bezier["Zone ≤15% (Clinically Accurate)"]
                resultBezier[5] += cega_bezier["Zone ≤30% (Likely Acceptable)"]
                resultBezier[6] += cega_bezier["Zone ≤50% (Clinically Safe)"]
                resultBezier[7] += cega_bezier["Zone >50% (Clinically Significant Error)"]

                resultLinear[0] += score_linear["RMSE Score"]
                resultLinear[1] += score_linear["EDM Score"]
                resultLinear[2] += score_linear["Peak Alignment Score"]
                resultLinear[3] += score_linear["Combined Score"]
                resultLinear[4] += cega_linear["Zone ≤15% (Clinically Accurate)"]
                resultLinear[5] += cega_linear["Zone ≤30% (Likely Acceptable)"]
                resultLinear[6] += cega_linear["Zone ≤50% (Clinically Safe)"]
                resultLinear[7] += cega_linear["Zone >50% (Clinically Significant Error)"]

                resultKNN[0] += score_knn["RMSE Score"]
                resultKNN[1] += score_knn["EDM Score"]
                resultKNN[2] += score_knn["Peak Alignment Score"]
                resultKNN[3] += score_knn["Combined Score"]
                resultKNN[4] += cega_knn["Zone ≤15% (Clinically Accurate)"]
                resultKNN[5] += cega_knn["Zone ≤30% (Likely Acceptable)"]
                resultKNN[6] += cega_knn["Zone ≤50% (Clinically Safe)"]
                resultKNN[7] += cega_knn["Zone >50% (Clinically Significant Error)"]

                resultBSpline[0] += score_bspline["RMSE Score"]
                resultBSpline[1] += score_bspline["EDM Score"]
                resultBSpline[2] += score_bspline["Peak Alignment Score"]
                resultBSpline[3] += score_bspline["Combined Score"]
                resultBSpline[4] += cega_bspline["Zone ≤15% (Clinically Accurate)"]
                resultBSpline[5] += cega_bspline["Zone ≤30% (Likely Acceptable)"]
                resultBSpline[6] += cega_bspline["Zone ≤50% (Clinically Safe)"]
                resultBSpline[7] += cega_bspline["Zone >50% (Clinically Significant Error)"]

                resultBasicCHIP[0] += score_chip["RMSE Score"]
                resultBasicCHIP[1] += score_chip["EDM Score"]
                resultBasicCHIP[2] += score_chip["Peak Alignment Score"]
                resultBasicCHIP[3] += score_chip["Combined Score"]
                resultBasicCHIP[4] += cega_chip["Zone ≤15% (Clinically Accurate)"]
                resultBasicCHIP[5] += cega_chip["Zone ≤30% (Likely Acceptable)"]
                resultBasicCHIP[6] += cega_chip["Zone ≤50% (Clinically Safe)"]
                resultBasicCHIP[7] += cega_chip["Zone >50% (Clinically Significant Error)"]

                
                totalAmount += 1




                # Open the file in append mode
                with open(file_name, 'a') as file:
                    # Write the text to the file
                    file.write("-------------------------\n")
                    file.write(str(fileList[i]) + "\n")
                    file.write(str(figureName) + "\n")
                    file.write(str(gapTime) + "\n")
                    file.write(str(numbers[j]) + "\n")
                    file.write(str(gapstart) + "\n")

                    file.write("Results for PCHIP \n")
                    file.write(str(score_pchip) + "\n")
                    for key, value in cega_pchip.items():
                        cega_a = str(key)
                        cega_b = str(value)
                        file.write(cega_b + "\n")

                    file.write("Results for Bezier \n")
                    file.write(str(score_bezier) + "\n")
                    for key, value in cega_bezier.items():
                        cega_a = str(key)
                        cega_b = str(value)
                        file.write(cega_b + "\n")
                    
                    
                    file.write("Results for Linear \n")
                    file.write(str(score_linear) + "\n")
                    for key, value in cega_linear.items():
                        cega_a = str(key)
                        cega_b = str(value)
                        file.write(cega_b + "\n")


with open(file_name, 'a') as file:
    # Write the text to the file
    file.write("-----------Summary--------------\n")
    file.write("Number of Gaps that have been created: " + str(totalAmount) + "\n")
    file.write("------------\n")
    file.write("PCHIP \n")
    for o in range(0,8):
        file.write(resultWords[o] + str(resultsPCHIP[o]/totalAmount) + "\n")
    file.write("------------\n")
    file.write("Bezier \n")
    for o in range(0,8):
        file.write(resultWords[o] + str(resultBezier[o]/totalAmount) + "\n")
    file.write("------------\n")
    file.write("Linear \n")
    for o in range(0,8):
        file.write(resultWords[o] + str(resultLinear[o]/totalAmount) + "\n")
    file.write("------------\n")
    file.write("BSpline \n")
    for o in range(0,8):
        file.write(resultWords[o] + str(resultBSpline[o]/totalAmount) + "\n")
    file.write("------------\n")
    file.write("KNN \n")
    for o in range(0,8):
        file.write(resultWords[o] + str(resultKNN[o]/totalAmount) + "\n")
    file.write("------------\n")
    file.write("Basic CHIP \n")
    for o in range(0,8):
        file.write(resultWords[o] + str(resultBasicCHIP[o]/totalAmount) + "\n")


    """
    file.write("RMSE: "  + str(totalRMSE/totalAmount) + "\n")
    file.write("EDM: "  + str(totalEDM/totalAmount) + "\n")
    file.write("Peak: "  + str(totalPeak/totalAmount) + "\n")
    file.write("Metric: "  + str(totalMetrik/totalAmount) + "\n")

    file.write("Clark15: "  + str(totalClark15/totalAmount) + "\n")
    file.write("Clark30: "  + str(totalClark30/totalAmount) + "\n")
    file.write("Clark50: "  + str(totalClark50/totalAmount) + "\n")
    file.write("Clark Over 50: "  + str(totalClarkMore/totalAmount) + "\n")

    file.write("Linear \n")
    file.write("RMSE: "  + str(totalRMSELinear/totalAmount) + "\n")
    file.write("EDM: "  + str(totalEDMLinear/totalAmount) + "\n")
    file.write("Peak: "  + str(totalPeakLinear/totalAmount) + "\n")
    file.write("Metric: "  + str(totalMetrikLinear/totalAmount) + "\n")

    file.write("Clark15: "  + str(totalClark15Linear/totalAmount) + "\n")
    file.write("Clark30: "  + str(totalClark30Linear/totalAmount) + "\n")
    file.write("Clark50: "  + str(totalClark50Linear/totalAmount) + "\n")
    file.write("Clark Over 50: "  + str(totalClarkMoreLinear/totalAmount) + "\n")
    """

    







