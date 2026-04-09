import pandas as pd
import glob
import os
import numpy as np

def get_latest_log():
    """Finds the most recent CSV file in the logs directory."""
    list_of_files = glob.glob('logs/*.csv')
    if not list_of_files:
        print("No CSV files found in the 'logs' directory. Play the game first!")
        return None
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file

def analyze_hypotheses(csv_path):
    print(f"\n--- Analyzing Data: {csv_path} ---\n")
    df = pd.read_csv(csv_path)

    # Create 'Epochs' to track each time the weather changes
    # This groups contiguous rows of the same weather together
    df['Weather_Shift'] = (df['Current_Weather'] != df['Current_Weather'].shift(1))
    df['Epoch_ID'] = df['Weather_Shift'].cumsum()

    # ==========================================
    # H4: THE BOREDOM ERROR (Action Bias)
    # ==========================================
    print(">> H4: Action Bias / Boredom Error (Stasis Phase)")
    grey_data = df[df['Current_Weather'] == 'GREY']
    if not grey_data.empty:
        # Check how many times the player moved (incurring a -0.1 penalty) during GREY
        grey_moves = grey_data[grey_data['Action_Taken'] != 'STAY']
        penalty_accumulated = grey_moves['Tick_Reward'].sum()
        time_in_grey = len(grey_data) / 10.0 # 10 ticks per second
        
        print(f"   Time spent in GREY (Stasis): {time_in_grey:.1f} seconds")
        print(f"   Unnecessary movements made: {len(grey_moves)}")
        print(f"   Points lost to boredom: {penalty_accumulated:.1f}")
        if len(grey_moves) > 0:
            print("   Verdict: H4 SUPPORTED. Human exhibited action bias instead of waiting.")
        else:
            print("   Verdict: Optimal play. Human stood perfectly still.")
    else:
        print("   (No GREY weather occurred in this session.)")

    # ==========================================
    # H2: PATCH ABANDONMENT (Stubbornness)
    # ==========================================
    print("\n>> H2: Patch Abandonment (Reaction to Volatility)")
    red_epochs = df[df['Current_Weather'] == 'RED']['Epoch_ID'].unique()
    abandonment_times = []
    
    for epoch in red_epochs:
        epoch_df = df[df['Epoch_ID'] == epoch]
        # Check if they started the RED phase inside a patch (Distance <= 2)
        if epoch_df.iloc[0]['Nearest_Bean_Dist'] <= 2:
            # Find the moment they walked away (Distance > 3)
            fled = epoch_df[epoch_df['Nearest_Bean_Dist'] > 3]
            if not fled.empty:
                ticks_to_flee = fled.iloc[0]['Ticks_In_Weather']
                abandonment_times.append(ticks_to_flee)

    if abandonment_times:
        avg_flee_ticks = np.mean(abandonment_times)
        print(f"   Average time to abandon patch in RED: {avg_flee_ticks} ticks ({avg_flee_ticks/10.0:.2f} sec)")
        print("   Verdict: Metric captured. Compare this against the RL agent (which usually stays until death).")
    else:
        print("   (Player never started a RED phase inside a patch, or never abandoned one).")

    # ==========================================
    # H3: RISK AVERSION
    # ==========================================
    print("\n>> H3: Risk Aversion (Rotten Penalties)")
    # Rotten beans give a -10 penalty. Factoring in movement, it might be -10.0 or -10.1.
    rotten_eaten = df[df['Tick_Reward'] <= -10]
    ripe_eaten = df[df['Tick_Reward'] >= 9.9]
    
    print(f"   Ripe Beans Harvested (+10): {len(ripe_eaten)}")
    print(f"   Rotten Beans Eaten (-10): {len(rotten_eaten)}")
    if len(rotten_eaten) <= 1:
        print("   Verdict: Strong Risk Aversion. Human successfully avoided negative variance.")
    else:
        print("   Verdict: Low Risk Aversion. Human fell for the high-variance trap.")

    # ==========================================
    # H1: SCHEMA MEMORY (Recovery Speed)
    # ==========================================
    print("\n>> H1: Schema Recovery (Safe Phase Return)")
    blue_epochs = df[df['Current_Weather'] == 'BLUE']['Epoch_ID'].unique()
    recovery_times = []
    
    for epoch in blue_epochs:
        # Skip the very first epoch of the game (Epoch 1)
        if epoch == 1: continue 
        
        epoch_df = df[df['Epoch_ID'] == epoch]
        # Find the first directional key press after returning to BLUE
        moves = epoch_df[epoch_df['Action_Taken'] != 'STAY']
        if not moves.empty:
            recovery_times.append(moves.iloc[0]['Ticks_In_Weather'])

    if recovery_times:
        avg_recovery = np.mean(recovery_times)
        print(f"   Average reaction time to resume foraging in BLUE: {avg_recovery:.1f} ticks ({avg_recovery/10.0:.2f} sec)")
        print("   Verdict: Metric captured. AI will show a much higher delay here due to catastrophic forgetting.")
    else:
        print("   (Not enough transitions back to BLUE to calculate recovery).")
    print("\n----------------------------------\n")


if __name__ == "__main__":
    latest_csv = get_latest_log()
    if latest_csv:
        analyze_hypotheses(latest_csv)