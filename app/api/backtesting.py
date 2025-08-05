"""
Backtesting API endpoints for strategy analysis
"""

from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime, date
import logging
import asyncio

from ..backtesting import (
    quick_backtest, 
    run_strategy_comparison,
    BacktestConfig,
    StrategyComparisonEngine,
    compare_all_strategies
)
from ..backtesting.performance_attribution import quick_attribution
from ..strategies.strategy_factory import StrategyFactory

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/backtesting", tags=["backtesting"])


class BacktestRequest(BaseModel):
    """Request model for backtesting"""
    strategy_name: str = Field(..., description="Strategy to backtest")
    universe: List[str] = Field(default=["AAPL", "MSFT", "GOOGL"], description="Stock universe")
    start_date: date = Field(default=date(2023, 1, 1), description="Backtest start date")
    end_date: date = Field(default=date(2024, 1, 1), description="Backtest end date")
    initial_capital: float = Field(default=100000.0, description="Initial capital")
    commission_rate: float = Field(default=0.001, description="Commission rate")
    max_position_pct: float = Field(default=0.05, description="Max position size %")


class ComparisonRequest(BaseModel):
    """Request model for strategy comparison"""
    strategies: List[str] = Field(..., description="Strategies to compare")
    universe: List[str] = Field(default=["AAPL", "MSFT", "GOOGL"], description="Stock universe")
    start_date: date = Field(default=date(2023, 1, 1), description="Backtest start date")
    end_date: date = Field(default=date(2024, 1, 1), description="Backtest end date")
    initial_capital: float = Field(default=100000.0, description="Initial capital")


class BacktestResponse(BaseModel):
    """Response model for backtesting results"""
    strategy_name: str
    total_return: float
    annual_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    insider_hit_rate: Optional[float] = None
    attribution_report: Optional[str] = None


# Background task storage (in production, use Redis or database)
_background_tasks = {}


@router.get("/strategies")
async def get_available_strategies():
    """Get list of available strategies for backtesting"""
    strategies = StrategyFactory.get_available_strategies()
    strategy_info = {}
    
    for strategy_name in strategies:
        info = StrategyFactory.get_strategy_info(strategy_name)
        strategy_info[strategy_name] = {
            "description": info.get("description", ""),
            "expected_performance": info.get("expected_performance", {}),
            "default_params": info.get("default_params", {})
        }
    
    return {
        "available_strategies": strategies,
        "strategy_details": strategy_info,
        "insider_strategies": [
            "insider_momentum_advanced",
            "insider_options_flow", 
            "insider_ml_predictor"
        ]
    }


@router.post("/run", response_model=BacktestResponse)
async def run_backtest(request: BacktestRequest):
    """
    Run backtest for a single strategy
    """
    logger.info(f"Running backtest for {request.strategy_name}")
    
    try:
        # Validate strategy
        available_strategies = StrategyFactory.get_available_strategies()
        if request.strategy_name not in available_strategies:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown strategy: {request.strategy_name}. Available: {available_strategies}"
            )
        
        # Run backtest
        results = await quick_backtest(
            strategy_name=request.strategy_name,
            universe=request.universe,
            start_date=request.start_date.strftime("%Y-%m-%d"),
            end_date=request.end_date.strftime("%Y-%m-%d")
        )
        
        # Generate attribution report for insider strategies
        attribution_report = None
        if "insider" in request.strategy_name:
            attribution_report = quick_attribution(results, request.strategy_name)
        
        return BacktestResponse(
            strategy_name=request.strategy_name,
            total_return=results.total_return,
            annual_return=results.annual_return,
            sharpe_ratio=results.sharpe_ratio,
            max_drawdown=results.max_drawdown,
            win_rate=results.win_rate,
            total_trades=results.total_trades,
            insider_hit_rate=results.insider_hit_rate if hasattr(results, 'insider_hit_rate') else None,
            attribution_report=attribution_report
        )
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/compare")
async def compare_strategies(request: ComparisonRequest, background_tasks: BackgroundTasks):
    """
    Compare multiple strategies (runs in background for large jobs)
    """
    logger.info(f"Comparing strategies: {request.strategies}")
    
    try:
        # Validate strategies
        available_strategies = StrategyFactory.get_available_strategies()
        invalid_strategies = [s for s in request.strategies if s not in available_strategies]
        
        if invalid_strategies:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown strategies: {invalid_strategies}. Available: {available_strategies}"
            )
        
        # Generate task ID
        task_id = f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Start background task
        background_tasks.add_task(
            _run_comparison_task,
            task_id,
            request.strategies,
            request.universe,
            request.start_date,
            request.end_date,
            request.initial_capital
        )
        
        return {
            "task_id": task_id,
            "status": "started",
            "message": f"Comparison of {len(request.strategies)} strategies started",
            "check_status_url": f"/backtesting/status/{task_id}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def _run_comparison_task(task_id: str, strategies: List[str], universe: List[str],
                              start_date: date, end_date: date, initial_capital: float):
    """Background task for strategy comparison"""
    
    try:
        _background_tasks[task_id] = {"status": "running", "progress": 0}
        
        # Create config
        config = BacktestConfig(
            start_date=datetime.combine(start_date, datetime.min.time()),
            end_date=datetime.combine(end_date, datetime.min.time()),
            initial_capital=initial_capital
        )
        
        # Run comparison
        results_dict = await run_strategy_comparison(strategies, universe, config)
        _background_tasks[task_id]["progress"] = 50
        
        # Analyze results
        comparison_engine = StrategyComparisonEngine()
        rankings = comparison_engine.rank_strategies(results_dict)
        report = comparison_engine.generate_comparison_report(rankings, results_dict)
        
        # Store results
        _background_tasks[task_id] = {
            "status": "completed",
            "progress": 100,
            "results": {
                "rankings": [
                    {
                        "strategy_name": r.strategy_name,
                        "rank": r.rank,
                        "overall_score": r.overall_score,
                        "recommendation": r.recommendation,
                        "strengths": r.strengths,
                        "weaknesses": r.weaknesses
                    }
                    for r in rankings
                ],
                "detailed_results": {
                    name: {
                        "total_return": results.total_return,
                        "annual_return": results.annual_return,
                        "sharpe_ratio": results.sharpe_ratio,
                        "max_drawdown": results.max_drawdown,
                        "win_rate": results.win_rate,
                        "total_trades": results.total_trades
                    }
                    for name, results in results_dict.items()
                },
                "report": report
            }
        }
        
        logger.info(f"Comparison task {task_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Comparison task {task_id} failed: {e}")
        _background_tasks[task_id] = {
            "status": "failed",
            "error": str(e)
        }


@router.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """Get status of background comparison task"""
    
    if task_id not in _background_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return _background_tasks[task_id]


@router.get("/quick-compare")
async def quick_strategy_comparison(
    strategies: str = Query(..., description="Comma-separated strategy names"),
    universe: str = Query("AAPL,MSFT,GOOGL", description="Comma-separated stock symbols"),
    days: int = Query(365, description="Number of days to backtest")
):
    """
    Quick comparison of strategies (synchronous, limited scope)
    """
    try:
        strategy_list = [s.strip() for s in strategies.split(",")]
        universe_list = [s.strip() for s in universe.split(",")]
        
        # Validate strategies
        available_strategies = StrategyFactory.get_available_strategies()
        invalid_strategies = [s for s in strategy_list if s not in available_strategies]
        
        if invalid_strategies:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown strategies: {invalid_strategies}"
            )
        
        # Quick comparison (simplified)
        end_date = datetime.now().date()
        start_date = datetime.now().date().replace(year=end_date.year - 1)
        
        comparison_results = {}
        
        for strategy_name in strategy_list:
            try:
                results = await quick_backtest(
                    strategy_name=strategy_name,
                    universe=universe_list,
                    start_date=start_date.strftime("%Y-%m-%d"),
                    end_date=end_date.strftime("%Y-%m-%d")
                )
                
                comparison_results[strategy_name] = {
                    "annual_return": results.annual_return,
                    "sharpe_ratio": results.sharpe_ratio,
                    "max_drawdown": results.max_drawdown,
                    "win_rate": results.win_rate,
                    "total_trades": results.total_trades
                }
                
            except Exception as e:
                logger.warning(f"Failed to backtest {strategy_name}: {e}")
                comparison_results[strategy_name] = {"error": str(e)}
        
        # Simple ranking
        valid_results = {k: v for k, v in comparison_results.items() if "error" not in v}
        
        if valid_results:
            # Rank by Sharpe ratio
            ranked_strategies = sorted(
                valid_results.items(),
                key=lambda x: x[1]["sharpe_ratio"],
                reverse=True
            )
            
            winner = ranked_strategies[0]
            
            return {
                "comparison_results": comparison_results,
                "recommended_strategy": winner[0],
                "winner_metrics": winner[1],
                "ranking": [{"strategy": k, "sharpe_ratio": v["sharpe_ratio"]} 
                           for k, v in ranked_strategies]
            }
        else:
            return {
                "comparison_results": comparison_results,
                "error": "No strategies completed successfully"
            }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Quick comparison failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/insider-analysis")
async def analyze_insider_strategies():
    """
    Specialized analysis for insider trading strategies
    """
    try:
        insider_strategies = [
            "insider_momentum_advanced",
            "insider_options_flow", 
            "insider_ml_predictor"
        ]
        
        # Generate comprehensive comparison report
        report = await compare_all_strategies(
            universe=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
            start_date="2023-01-01",
            end_date="2024-01-01"
        )
        
        return {
            "insider_strategies": insider_strategies,
            "analysis_report": report,
            "recommendations": {
                "best_for_alpha": "insider_ml_predictor",
                "best_for_sharpe": "insider_momentum_advanced", 
                "best_for_consistency": "insider_options_flow",
                "best_overall": "insider_ml_predictor"
            }
        }
        
    except Exception as e:
        logger.error(f"Insider analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Health check endpoint
@router.get("/health")
async def backtesting_health():
    """Health check for backtesting service"""
    return {
        "status": "healthy",
        "available_strategies": len(StrategyFactory.get_available_strategies()),
        "insider_strategies_available": True,
        "background_tasks": len(_background_tasks)
    }