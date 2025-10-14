"""Standalone test for trade optimizer without external dependencies."""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.trade_optimizer import TradeOptimizer


def test_direct_swap_optimization():
    """Test that optimizer finds direct swap opportunities."""
    print("\n" + "="*70)
    print("TEST 1: Direct Swap Optimization (ETH → BTC)")
    print("="*70)

    optimizer = TradeOptimizer()

    to_sell = {'ETH': 500.0}
    to_buy = {'BTC': 500.0}
    available_pairs = {
        'BTC-ETH': True,
        'ETH-USD': True,
        'BTC-USD': True,
    }

    trades = optimizer.calculate_optimal_trades(
        to_sell=to_sell,
        to_buy=to_buy,
        available_pairs=available_pairs,
        prefer_direct=True
    )

    print(f"\n✓ Found {len(trades)} trade(s)")
    for i, trade in enumerate(trades, 1):
        print(f"\nTrade {i}:")
        print(f"  From: {trade['from_asset']}")
        print(f"  To: {trade['to_asset']}")
        print(f"  Product: {trade['product_id']}")
        print(f"  Value: ${trade['value_usd']:.2f}")
        print(f"  Direct: {trade['is_direct']}")
        print(f"  Reason: {trade['reason']}")

    assert len(trades) == 1, f"Expected 1 trade, got {len(trades)}"
    assert trades[0]['is_direct'] == True, "Expected direct trade"
    print("\n✅ TEST PASSED: Direct swap detected correctly")


def test_no_direct_pair_routes_through_usd():
    """Test USD routing when no direct pair exists."""
    print("\n" + "="*70)
    print("TEST 2: USD Routing (SOL → BTC, no direct pair)")
    print("="*70)

    optimizer = TradeOptimizer()

    to_sell = {'SOL': 500.0}
    to_buy = {'BTC': 500.0}
    available_pairs = {
        'SOL-USD': True,
        'BTC-USD': True,
        # No SOL-BTC pair
    }

    trades = optimizer.calculate_optimal_trades(
        to_sell=to_sell,
        to_buy=to_buy,
        available_pairs=available_pairs,
        prefer_direct=True
    )

    print(f"\n✓ Found {len(trades)} trade(s)")
    for i, trade in enumerate(trades, 1):
        print(f"\nTrade {i}:")
        print(f"  Type: {trade['type']}")
        print(f"  From: {trade['from_asset']}")
        print(f"  To: {trade['to_asset']}")
        print(f"  Product: {trade['product_id']}")
        print(f"  Value: ${trade['value_usd']:.2f}")
        print(f"  Direct: {trade['is_direct']}")

    assert len(trades) == 2, f"Expected 2 trades, got {len(trades)}"
    assert all(not t['is_direct'] for t in trades), "Expected USD-routed trades"
    print("\n✅ TEST PASSED: USD routing works correctly")


def test_complex_multi_asset_rebalance():
    """Test complex scenario with multiple assets."""
    print("\n" + "="*70)
    print("TEST 3: Complex Multi-Asset Rebalancing")
    print("="*70)

    optimizer = TradeOptimizer()

    to_sell = {
        'BTC': 300.0,
        'ETH': 200.0
    }
    to_buy = {
        'SOL': 250.0,
        'MATIC': 250.0
    }
    available_pairs = {
        'BTC-USD': True,
        'ETH-USD': True,
        'SOL-USD': True,
        'MATIC-USD': True,
        'ETH-BTC': True,
        'SOL-ETH': True,
    }

    trades = optimizer.calculate_optimal_trades(
        to_sell=to_sell,
        to_buy=to_buy,
        available_pairs=available_pairs,
        prefer_direct=True
    )

    direct_trades = [t for t in trades if t.get('is_direct', False)]
    usd_trades = [t for t in trades if not t.get('is_direct', False)]

    print(f"\n✓ Total trades: {len(trades)}")
    print(f"  Direct trades: {len(direct_trades)}")
    print(f"  USD-routed trades: {len(usd_trades)}")

    print("\nDirect trades:")
    for trade in direct_trades:
        print(f"  {trade['from_asset']} → {trade['to_asset']}: ${trade['value_usd']:.2f}")

    print("\nUSD-routed trades:")
    for trade in usd_trades:
        print(f"  {trade['from_asset']} → {trade['to_asset']}: ${trade['value_usd']:.2f}")

    assert len(direct_trades) > 0, "Expected some direct trades"
    print("\n✅ TEST PASSED: Complex rebalancing handled correctly")


def test_fee_savings_demonstration():
    """Demonstrate fee savings from optimization."""
    print("\n" + "="*70)
    print("TEST 4: Fee Savings Calculation")
    print("="*70)

    optimizer = TradeOptimizer()

    to_sell = {'ETH': 500.0}
    to_buy = {'BTC': 500.0}
    available_pairs = {
        'BTC-ETH': True,
        'ETH-USD': True,
        'BTC-USD': True,
    }

    # Optimized approach
    trades_optimized = optimizer.calculate_optimal_trades(
        to_sell=to_sell,
        to_buy=to_buy,
        available_pairs=available_pairs,
        prefer_direct=True
    )

    # Naive approach
    trades_naive = optimizer.calculate_optimal_trades(
        to_sell=to_sell,
        to_buy=to_buy,
        available_pairs=available_pairs,
        prefer_direct=False
    )

    fee_rate = 0.006  # 0.6% fee per trade
    opt_fee = len(trades_optimized) * 500.0 * fee_rate
    naive_fee = len(trades_naive) * 500.0 * fee_rate
    savings = naive_fee - opt_fee
    savings_pct = (savings / naive_fee) * 100

    print(f"\nScenario: Rebalance $500 from ETH to BTC")
    print(f"Assumed fee rate: {fee_rate*100}%")
    print()
    print(f"OPTIMIZED APPROACH:")
    print(f"  Trades: {len(trades_optimized)}")
    print(f"  Fees: ${opt_fee:.2f}")
    print()
    print(f"NAIVE APPROACH:")
    print(f"  Trades: {len(trades_naive)}")
    print(f"  Fees: ${naive_fee:.2f}")
    print()
    print(f"💰 SAVINGS: ${savings:.2f} ({savings_pct:.0f}%)")

    assert savings > 0, "Expected cost savings"
    assert len(trades_optimized) < len(trades_naive), "Expected fewer trades"
    print("\n✅ TEST PASSED: Optimization reduces fees")


def test_reverse_pair_detection():
    """Test that optimizer handles pairs in both directions."""
    print("\n" + "="*70)
    print("TEST 5: Reverse Pair Detection (ETH-BTC vs BTC-ETH)")
    print("="*70)

    optimizer = TradeOptimizer()

    to_sell = {'BTC': 500.0}
    to_buy = {'ETH': 500.0}

    # Pair exists as ETH-BTC (reverse)
    available_pairs = {
        'ETH-BTC': True,  # Note: not BTC-ETH
        'BTC-USD': True,
        'ETH-USD': True,
    }

    trades = optimizer.calculate_optimal_trades(
        to_sell=to_sell,
        to_buy=to_buy,
        available_pairs=available_pairs,
        prefer_direct=True
    )

    print(f"\n✓ Found {len(trades)} trade(s)")
    print(f"  Product: {trades[0]['product_id']}")
    print(f"  Direct: {trades[0]['is_direct']}")

    assert len(trades) == 1, "Expected 1 trade"
    assert trades[0]['is_direct'] == True, "Expected direct trade"
    assert trades[0]['product_id'] == 'ETH-BTC', "Expected ETH-BTC pair"
    print("\n✅ TEST PASSED: Reverse pair detection works")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("TRADE OPTIMIZER TEST SUITE")
    print("="*70)

    tests = [
        test_direct_swap_optimization,
        test_no_direct_pair_routes_through_usd,
        test_complex_multi_asset_rebalance,
        test_fee_savings_demonstration,
        test_reverse_pair_detection,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"\n❌ TEST FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"\n❌ TEST ERROR: {e}")
            failed += 1

    print("\n" + "="*70)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("="*70)

    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
