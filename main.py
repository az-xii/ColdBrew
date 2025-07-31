#! venv/bin/python
from tradelab import TradeLab
from coldbrew import ColdBrewGeneration
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import json
import glob

def get_training_period():
    """Get training period within last 60 days"""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=59)  # Stay within 60-day window
    return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')

def plot_generation_performance(generation, env, gen_number, population_size):
    """Plot all ColdBrews vs market performance"""
    plt.figure(figsize=(15, 10))
    
    # Plot market benchmark (S&P 500)
    benchmark = env.get_benchmark_performance()
    if benchmark:
        plt.plot(benchmark, label='S&P 500 Benchmark', color='black', linewidth=3, alpha=0.7)
    
    # Plot each ColdBrew's portfolio value over time
    colors = plt.cm.tab20(np.linspace(0, 1, population_size))  # 20 different colors
    for i, coldbrew in enumerate(generation.coldbrews):
        portfolio_history = coldbrew.get_portfolio_history()
        if len(portfolio_history) > 1:
            plt.plot(portfolio_history, 
                    color=colors[i], 
                    alpha=0.6, 
                    linewidth=1,
                    label=f'ColdBrew-{i+1}' if i < 5 else "")  # Only label top 5
    
    # Highlight the top 3 performers
    top_3 = generation.get_top_performers(3)
    highlight_colors = ['gold', 'silver', '#CD7F32']  # Gold, silver, bronze
    
    for i, champion in enumerate(top_3):
        portfolio_history = champion.get_portfolio_history()
        if len(portfolio_history) > 1:
            plt.plot(portfolio_history, 
                    color=highlight_colors[i],
                    linewidth=3,
                    label=f'#{i+1} Champion (${champion.portfolio_value[-1]:.0f})')
    
    plt.title(f'Generation {gen_number} Performance\n{generation.training_period}', fontsize=16)
    plt.xlabel('Trading Intervals (5-min)', fontsize=12)
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    os.makedirs('generation_plots', exist_ok=True)
    plt.savefig(f'generation_plots/generation_{gen_number:03d}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Generation {gen_number} plot saved!")

def save_champion_brains(generation, gen_number):
    """Save the DNA of top 4 performers for posterity"""
    top_4 = generation.get_top_performers(4)
    
    # Create champions directory
    os.makedirs('champion_brains', exist_ok=True)
    
    champions_data = {
        'generation': gen_number,
        'timestamp': datetime.now().isoformat(),
        'training_period': generation.training_period,
        'champions': []
    }
    
    for rank, champion in enumerate(top_4, 1):
        champion_info = {
            'rank': rank,
            'dna': champion.dna,
            'final_portfolio': champion.portfolio_value[-1],
            'fitness_score': champion.calculate_fitness(),
            'total_trades': len(champion.trades),
            'prediction_accuracy': len([1 for p, a in zip(champion.predictions, champion.actual_movements) 
                                      if (p > 0 and a > 0) or (p < 0 and a < 0)]) / len(champion.predictions) if champion.predictions else 0
        }
        champions_data['champions'].append(champion_info)
    
    # Save individual champion files
    for rank, champion in enumerate(top_4, 1):
        champion_file = f'champion_brains/gen_{gen_number:03d}_rank_{rank}_brain.json'
        individual_data = {
            'generation': gen_number,
            'rank': rank,
            'dna': champion.dna,
            'performance': {
                'final_portfolio': champion.portfolio_value[-1],
                'fitness_score': champion.calculate_fitness(),
                'roi_percent': ((champion.portfolio_value[-1] / champion.initial_funds) - 1) * 100
            },
            'training_period': generation.training_period,
            'saved_at': datetime.now().isoformat()
        }
        
        with open(champion_file, 'w') as f:
            json.dump(individual_data, f, indent=2)
    
    # Save generation summary
    summary_file = f'champion_brains/generation_{gen_number:03d}_champions.json'
    with open(summary_file, 'w') as f:
        json.dump(champions_data, f, indent=2)
    
    print(f"🧠 Top 4 champion brains saved from Generation {gen_number}")
    print(f"   🥇 Champion: ${top_4[0].portfolio_value[-1]:.0f} (ROI: {((top_4[0].portfolio_value[-1]/1000)-1)*100:.1f}%)")
    return champions_data

def load_champion_brain(champion_file):
    """Load a champion's brain from file"""
    try:
        with open(champion_file, 'r') as f:
            champion_data = json.load(f)
        
        from coldbrew import ColdBrew
        champion = ColdBrew(
            dna=champion_data['dna'],
            initial_funds=1000,
            symbols=['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
        )
        
        print(f"🧠 Loaded champion from Generation {champion_data['generation']}")
        print(f"   Original Performance: ${champion_data['performance']['final_portfolio']:.0f}")
        print(f"   ROI: {champion_data['performance']['roi_percent']:.1f}%")
        
        return champion
        
    except FileNotFoundError:
        print(f"❌ Champion file not found: {champion_file}")
        return None
    except Exception as e:
        print(f"❌ Error loading champion: {e}")
        return None

def create_all_stars_tournament():
    """Create a tournament with champions from different generations"""
    print("🏆 Creating All-Stars Tournament!")
    
    # Find all champion files
    champion_files = glob.glob('champion_brains/gen_*_rank_1_brain.json')
    
    if len(champion_files) < 2:
        print("❌ Need at least 2 generations to run tournament")
        return
    
    # Load champions
    champions = []
    for file in sorted(champion_files)[:10]:  # Top 10 generations
        champion = load_champion_brain(file)
        if champion:
            champions.append(champion)
    
    print(f"🥊 Tournament ready with {len(champions)} champions!")
    
    # TODO: Run them all in the same market period and see who wins
    # This would be a fun addition for later!

def main():
    """Main evolution loop"""
    # Configuration
    generation_number = 1
    population_size = int(input("Population size :"))
    initial_funds = 1000
    symbols = [
        'AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMD', 'AVGO', 'PLTR', 'QCOM',
        'BRK-B', 'HON', 'MMM', 'MDT',
        'SPY', 'QQQ', 'IWM', 'SOXL', 'SOXS',
        'BHP', 'BP', 'HSBC', 'AZN', 'PDD'
    ]
    
    # Training parameters
    training_intervals = int(input("Intervals :"))
    
    print("🔥 ColdBrew Evolution Program vDunno")
    print("⚠️  TRAINING MODE ONLY - No real money involved!")
    print("🎯 Target: 4 years of evolution until ready for live trading\n")
    print(f"📊 Population: {population_size} ColdBrews per generation")
    print(f"⏰ Training: {training_intervals} intervals @ 5 minutes each (3+ weeks of market data)")
    print(f"📈 Symbols: {symbols}\n")
    
    # Initialize population
    generation = ColdBrewGeneration(
        population_size=population_size,
        initial_funds=initial_funds,
        symbols=symbols
    )
    
    try:
        # Evolution loop (TRAINING ONLY!)
        while True:
            print(f"🧬 Generation {generation_number} brewing...")
            
            # Get fresh real-world training data
            start_date, end_date = get_training_period()
            generation.training_period = f"{start_date} to {end_date}"
            
            # Create environment with real market data
            try:
                env = TradeLab(
                    symbols=symbols,
                    mode='historical',
                    start_date=start_date,
                    end_date=end_date
                )
            except Exception as e:
                print(f"❌ Failed to load market data: {e}")
                print("🔄 Trying different date range...")
                continue
            
            print(f"📚 Training period: {start_date} to {end_date}")
            
            # Training loop - fixed 1500 intervals for proper education
            trading_intervals = 0
            
            while trading_intervals < training_intervals and env.step():
                # All ColdBrews make trading decisions
                generation.trade_step(env)
                trading_intervals += 1
                
                # Progress indicator every 100 intervals
                if trading_intervals % 100 == 0:
                    print(f"   ⏳ Interval {trading_intervals}/{training_intervals}")
            
            print(f"✅ Training complete: {trading_intervals} intervals ({trading_intervals * 5 / 60:.1f} hours of market data)")
            
            # Update final portfolio values
            for brew in generation.coldbrews:
                if brew.portfolio_value:
                    brew.final_portfolio = brew.portfolio_value[-1]
            
            # Create visualization
            plot_generation_performance(generation, env, generation_number, population_size)
            
            # Show performance report
            feedback = generation.get_feedback()
            print(f"📊 Generation {generation_number} Results:")
            print(f"    🏆 Best Fitness: {feedback['best_fitness']:.2f}")
            print(f"    💰 Average Profit: ${feedback['avg_profit']:.2f}")
            print(f"    🎯 Top Prediction Accuracy: {feedback['best_accuracy']:.1%}")
            print(f"    📈 Market Conditions: {feedback['market_type']}")
            
            # Show top performers
            top_4 = generation.get_top_performers(4)
            print(f"    🥇 Champion: ${top_4[0].portfolio_value[-1]:.0f} (Fitness: {top_4[0].calculate_fitness():.1f})")
            if len(top_4) > 1:
                print(f"    🥈 Runner-up: ${top_4[1].portfolio_value[-1]:.0f} (Fitness: {top_4[1].calculate_fitness():.1f})")
            if len(top_4) > 2:
                print(f"    🥉 Third: ${top_4[2].portfolio_value[-1]:.0f} (Fitness: {top_4[2].calculate_fitness():.1f})")
            if len(top_4) > 3:
                print(f"    🏅 Fourth: ${top_4[3].portfolio_value[-1]:.0f} (Fitness: {top_4[3].calculate_fitness():.1f})")
            
            
            # Evolution (breed next generation)
            print("🔬 Evolution in progress...")
            generation.cull(generation_number)
            generation_number += 1
            
            # Save progress every 10 generations
            if generation_number % 10 == 0:
                os.makedirs('generation_plots', exist_ok=True)
                checkpoint_file = f'checkpoints/gen_{generation_number}.json'
                generation.save_evolution_progress(checkpoint_file)
                print(f"💾 Checkpoint saved: {checkpoint_file}")
            
            print(f"➡️  Generation {generation_number} created\n")
            print("-" * 60)
            
            # Optional: Pause for review (uncomment if you want manual control)
            # input("Press Enter for next generation or Ctrl+C to stop...")
    
    except KeyboardInterrupt:
        print("\n\n🛑 Evolution stopped by user")
        print(f"📊 Final Stats:")
        print(f"   Generations completed: {generation_number - 1}")
        
        if generation.coldbrews:
            final_champion = generation.get_top_performers(1)[0]
            print(f"   🏆 Best ColdBrew: ${final_champion.portfolio_value[-1]:.0f}")
            print(f"   🧠 Champion Fitness: {final_champion.calculate_fitness():.1f}")
        
        print("🚀 Evolution data saved in generation_plots/ and checkpoint files")
        print("💡 When ready for live trading, manually deploy the champion!")

if __name__ == "__main__":
    main()