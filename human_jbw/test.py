from wrapper import ContextualVolatilityWrapper, BeanState

def run_headless_test():
    print("--- STARTING HEADLESS LOGIC TEST ---\n")
    
    # 1. Initialize the wrapper WITHOUT the real JBW environment
    test_env = ContextualVolatilityWrapper(jbw_env=None)
    obs, info = test_env.reset()
    
    print(f"Starting Weather: {info['weather']}")
    print(f"P_ripen: {info['P_ripen']}, P_rot: {info['P_rot']}\n")
    
    # 2. Manually inject 3 Green beans to watch them evolve
    test_env.active_beans[(0, 0)] = BeanState.GREEN
    test_env.active_beans[(5, 5)] = BeanState.GREEN
    test_env.active_beans[(10, 10)] = BeanState.GREEN
    
    print(f"Initial Beans: {test_env.active_beans}\n")

    # 3. Run a simulation loop for 100 steps
    weather_changes = 0
    last_weather = info['weather']  # Initialize before loop
    
    for step in range(1, 101):
        # We pass '0' as a dummy action since we don't have a real agent yet
        obs, reward, done, info = test_env.step(0)
        
        # Track if the weather changed
        current_weather = info['weather']
        if current_weather != last_weather:
            print(f"Step {step:3d}: ⚡ WEATHER CHANGED! {last_weather} -> {current_weather}")
            weather_changes += 1
            last_weather = current_weather
        
        # Print the state of the beans every 10 steps to avoid spamming the console
        if step % 10 == 0:
            print(f"Step {step:3d} | Weather: {current_weather:4s} | Beans: {test_env.active_beans}")

    print("\n--- TEST COMPLETE ---")
    print(f"Total Weather Shifts: {weather_changes}")
    print(f"Final Beans: {test_env.active_beans}")
    
    # Verify context-based probabilities were different
    print(f"\nContext Probabilities (BLUE): {test_env.context_probs[test_env.current_weather]}")

if __name__ == "__main__":
    run_headless_test()