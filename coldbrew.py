import numpy as np
import random
import json
from datetime import datetime
from collections import defaultdict

class ColdBrew:
    def __init__(self, dna=None, initial_funds=1000, symbols=None):
        self.initial_funds = initial_funds
        self.symbols = symbols or []
        self.dna = dna or self._generate_random_dna()
        self.reset_portfolio()
        
    def _generate_random_dna(self):
        """Generate random neural network weights with trading bias"""
        return {
            'weights_prediction': (np.random.randn(10) * 1.5).tolist(),  # More volatile initial weights
            'weights_strategy': [
                random.uniform(-0.5, 0.5),  # trade_threshold modifier
                random.uniform(0.1, 2.0),   # position_size multiplier
                0, 0, 0  # Reserved for future use
            ],
            'risk_appetite': random.uniform(0.3, 0.9),
            'trade_frequency': random.randint(1, 10),
            'min_trade_size': random.uniform(0.05, 0.2)  # Minimum position size
        }
    
    def reset_portfolio(self):
        """Reset portfolio to initial state"""
        self.cash = self.initial_funds
        self.holdings = {symbol: 0 for symbol in self.symbols}
        self.portfolio_value = [self.initial_funds]
        self.trades = []
        self.predictions = []
        self.actual_movements = []
        self.final_portfolio = self.initial_funds
    

    def predict_movement(self, state):
        """Predict market movement with NaN protection"""
        # Replace NaN values before feature calculation
        returns = state['returns'] if not np.isnan(state['returns']) else random.uniform(-0.01, 0.01)
        ma_10 = state['ma_10'] if not np.isnan(state['ma_10']) else state['close']
        ma_50 = state['ma_50'] if not np.isnan(state['ma_50']) else state['close']
        rsi = state['rsi'] if not np.isnan(state['rsi']) else random.uniform(30, 70)
        
        features = np.array([
            returns,
            ma_10,
            ma_50,
            rsi,
            state['volume'],
            state['close'] - state['open'],
            state['high'] - state['low'],
            1 if state['close'] > ma_10 else -1,  # More pronounced signals
            1 if state['close'] > ma_50 else -1,
            (rsi - 50) / 50  # Normalized RSI
        ])
        
        weights = np.array(self.dna['weights_prediction'])
        prediction = np.dot(features, weights)
        return np.tanh(prediction * 2)  # More aggressive scaling
    
    def make_trade_decision(self, state, prediction):
        """More permissive trade decision logic"""
        symbol = state['symbol']
        current_price = state['close']
        
        # Dynamic thresholds based on volatility
        price_range = state['high'] - state['low']
        volatility_adjusted_threshold = 0.1 * (price_range / state['close'])
        
        buy_threshold = volatility_adjusted_threshold + (self.dna['weights_strategy'][0] * 0.05)
        sell_threshold = -volatility_adjusted_threshold + (self.dna['weights_strategy'][0] * 0.05)
        
        # Ensure minimum trade size
        position_size = max(
            self.dna['min_trade_size'],
            abs(prediction) * self.dna['risk_appetite'] * self.dna['weights_strategy'][1]
        )
        
        # Calculate shares - ensure at least 1 share can be traded
        max_affordable = max(1, int((self.cash * position_size) / current_price)) if current_price > 0 else 0
        current_holdings = self.holdings[symbol]
        
        # Trade decision
        if prediction > buy_threshold and max_affordable > 0:
            shares = min(
                max_affordable,
                max(1, int(self.dna['trade_frequency'] * position_size))
            )
            return 'buy', shares
        
        elif prediction < sell_threshold and current_holdings > 0:
            shares = min(
                current_holdings,
                max(1, int(self.dna['trade_frequency'] * position_size))
            )
            return 'sell', shares
        
        return 'hold', 0

    def update_portfolio(self, trade, current_prices):
        """Update portfolio based on trade execution"""
        symbol = trade['symbol']
        action = trade['action']
        shares = trade['shares']
        cost = trade['cost']
        
        if action == 'buy':
            self.cash -= cost
            self.holdings[symbol] += shares
        elif action == 'sell':
            self.cash += cost
            self.holdings[symbol] -= shares
        
        # Update portfolio value
        holdings_value = sum(
            self.holdings[sym] * current_prices.get(sym, 0)
            for sym in self.symbols
        )
        self.portfolio_value.append(self.cash + holdings_value)
    
    def get_portfolio_history(self):
        """Get normalized portfolio history"""
        return [v / self.initial_funds for v in self.portfolio_value]
    
    def calculate_fitness(self):
        """Calculate fitness score based on multiple factors"""
        # 1. Profit (40%)
        profit_score = (self.portfolio_value[-1] / self.initial_funds - 1) * 100
        
        # 2. Prediction Accuracy (30%)
        if self.predictions:
            correct = sum(
                1 for p, a in zip(self.predictions, self.actual_movements) 
                if (p > 0 and a > 0) or (p < 0 and a < 0)
            )
            accuracy_score = (correct / len(self.predictions)) * 100
        else:
            accuracy_score = 50  # Default if no predictions
        
        # 3. Risk Management (20%)
        if len(self.portfolio_value) > 1:
            returns = np.diff(self.portfolio_value) / self.portfolio_value[:-1]
            volatility = np.std(returns) * 100 if len(returns) > 0 else 0
            risk_score = max(0, 100 - volatility)
        else:
            risk_score = 50
        
        # 4. Market Timing (10%)
        if self.trades:
            trade_profits = []
            for i, trade in enumerate(self.trades):
                if i < len(self.portfolio_value) - 1:
                    profit = (self.portfolio_value[i+1] - self.portfolio_value[i]) / self.portfolio_value[i]
                    trade_profits.append(profit)
            timing_score = np.mean(trade_profits) * 100 if trade_profits else 0
        else:
            timing_score = 0
        
        profit_score = (self.portfolio_value[-1] / self.initial_funds - 1) * 100
        
        # Trading activity bonus (up to 30% of score)
        trade_activity = min(1.0, len(self.trades) / 30) * 30  # Normalized to 0-30
        
        # Modified weighted score
        return (
            0.4 * profit_score +
            0.3 * (accuracy_score if self.predictions else 50) +
            0.2 * risk_score +
            0.1 * timing_score +
            trade_activity  # Add trading activity bonus
        )

class ColdBrewGeneration:
    def __init__(self, population_size=50, initial_funds=1000, symbols=None):
        self.population_size = population_size
        self.initial_funds = initial_funds
        self.symbols = symbols or []
        self.coldbrews = [ColdBrew(initial_funds=initial_funds, symbols=symbols) 
                         for _ in range(population_size)]
        self.training_period = ""
    
    def trade_step(self, env):
        """Execute one trading interval with validation"""
        current_prices = {}
        
        for symbol in self.symbols:
            state = env.get_current_state(symbol)
            if state is None:
                continue
                
            current_prices[symbol] = state['close']
            
            for brew in self.coldbrews:
                # Skip if we can't afford even one share
                if state['close'] > brew.cash * 0.9:  # Leave 10% buffer
                    continue
                    
                prediction = brew.predict_movement(state)
                action, shares = brew.make_trade_decision(state, prediction)
                
                if action != 'hold' and shares > 0:
                    trade = env.execute_trade(symbol, action, shares, state['close'])
                    brew.trades.append(trade)
                    brew.update_portfolio(trade, current_prices)
                
                brew.predictions.append(prediction)
        
        # Move to next interval and record actual movements
        if env.step():
            for brew in self.coldbrews:
                for symbol in self.symbols:
                    next_state = env.get_current_state(symbol)
                    if next_state and symbol in current_prices:
                        actual_move = (next_state['close'] - current_prices[symbol]) / current_prices[symbol]
                        brew.actual_movements.append(actual_move)
            env.current_step -= 1

    def get_top_performers(self, n=3):
        """Return top n performers based on fitness"""
        ranked = sorted(self.coldbrews, 
                       key=lambda x: x.calculate_fitness(), 
                       reverse=True)
        return ranked[:n]
    
    def get_feedback(self):
        """Get generation performance statistics"""
        profits = [c.portfolio_value[-1] - c.initial_funds for c in self.coldbrews]
        accuracies = []
        
        for brew in self.coldbrews:
            if brew.predictions and brew.actual_movements:
                min_len = min(len(brew.predictions), len(brew.actual_movements))
                if min_len > 0:
                    correct = sum(
                        1 for p, a in zip(brew.predictions[:min_len], brew.actual_movements[:min_len]) 
                        if (p > 0 and a > 0) or (p < 0 and a < 0)
                    )
                    accuracies.append(correct / min_len)
        
        # Determine market type
        all_movements = []
        for brew in self.coldbrews:
            all_movements.extend(brew.actual_movements)
        
        avg_move = np.mean(all_movements) if all_movements else 0
        market_type = "Bullish" if avg_move > 0 else "Bearish"
        
        return {
            'best_fitness': max(c.calculate_fitness() for c in self.coldbrews),
            'avg_profit': np.mean(profits),
            'best_accuracy': max(accuracies) if accuracies else 0,
            'market_type': f"{market_type} ({avg_move:.4f}/interval)"
        }
    
    def cull(self, generation_number):
        """Cull bottom performers, breed top performers to create new generation"""
        # Rank all coldbrews by fitness
        ranked = sorted(self.coldbrews, 
                       key=lambda x: x.calculate_fitness(), 
                       reverse=True)
        
        # Top 20% become breeding champions
        elites = int(0.2 * self.population_size)
        breeds = self.population_size - elites
        champions = ranked[:elites]
        
        # Create new generation
        new_generation = []
        
        # 1. Carry over champions unchanged (elitism)
        for champion in champions:
            new_champion = ColdBrew(
                dna=champion.dna.copy(),
                initial_funds=self.initial_funds,
                symbols=self.symbols
            )
            new_generation.append(new_champion)
        
        # 2. Breed new agents from champions
        for _ in range(breeds):
            parent1, parent2 = random.sample(champions, 2)
            child_dna = self._crossover(parent1.dna, parent2.dna, generation_number)
            new_generation.append(ColdBrew(
                dna=child_dna,
                initial_funds=self.initial_funds,
                symbols=self.symbols
            ))
        
        self.coldbrews = new_generation
    
    def _crossover(self, dna1, dna2, generation_number):
        """Combine two DNA sets with mutation"""
        child_dna = {}
        mutation_rate = max(0.4, 0.2 * np.exp(-generation_number / 1.204))
        for key in dna1.keys():
            if isinstance(dna1[key], list):
                # Crossover list elements
                child_vec = []
                for i in range(len(dna1[key])):
                    if random.random() > 0.5:
                        base = dna1[key][i]
                    else:
                        base = dna2[key][i]
                    
                    # Apply mutation
                    if random.random() < mutation_rate:  
                        base += random.gauss(0, max(0.1, mutation_rate))
                    child_vec.append(base)
                child_dna[key] = child_vec
            else:
                # Crossover single values
                if random.random() > 0.5:
                    base = dna1[key]
                else:
                    base = dna2[key]
                
                # Apply mutation
                if random.random() < 0.2:
                    if isinstance(base, (int, float)):
                        base = min(max(base * random.uniform(0.8, 1.2), 0.01), 0.95)
                child_dna[key] = base
        return child_dna
    
    def early_graduation_check(self):
        """No early graduation - always complete full intervals"""
        return False  # Disabled for proper market education
    
    def save_evolution_progress(self, filename):
        """Save current generation's DNA for future use"""
        evolution_data = {
            'timestamp': datetime.now().isoformat(),
            'generation': [{
                'dna': brew.dna,
                'fitness': brew.calculate_fitness(),
                'final_portfolio': brew.portfolio_value[-1]
            } for brew in self.coldbrews]
        }
        
        with open(filename, 'w') as f:
            json.dump(evolution_data, f, indent=2)
        
        print(f"💾 Evolution progress saved to {filename}")
