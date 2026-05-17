from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import text
from typing import Optional
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))
from backend.models.database import get_db

router = APIRouter(prefix="/api/coe", tags=["COE"])


@router.get("/latest")
def get_latest_premiums(db: Session = Depends(get_db)):
    """Get the most recent COE premium for each category."""
    result = db.execute(text("""
        SELECT vehicle_class, premium, quota, bids_received, month
        FROM coe_results
        WHERE month = (SELECT MAX(month) FROM coe_results)
        ORDER BY vehicle_class
    """)).fetchall()

    return {
        "data": [
            {
                "category": row[0],
                "premium": row[1],
                "quota": row[2],
                "bids_received": row[3],
                "month": str(row[4])[:7],
            }
            for row in result
        ]
    }


@router.get("/history")
def get_history(
    category: Optional[str] = Query(None, description="e.g. 'Category A'"),
    start_year: Optional[int] = Query(None, description="e.g. 2020"),
    end_year: Optional[int] = Query(None, description="e.g. 2024"),
    db: Session = Depends(get_db),
):
    """Get full COE history with optional filters."""
    query = """
        SELECT vehicle_class, premium, quota, bids_received, bids_success, month
        FROM coe_results
        WHERE 1=1
    """
    params = {}

    if category:
        query += " AND vehicle_class = :category"
        params["category"] = category

    if start_year:
        query += " AND strftime('%Y', month) >= :start_year"
        params["start_year"] = str(start_year)

    if end_year:
        query += " AND strftime('%Y', month) <= :end_year"
        params["end_year"] = str(end_year)

    query += " ORDER BY month, vehicle_class"

    result = db.execute(text(query), params).fetchall()

    return {
        "count": len(result),
        "data": [
            {
                "category": row[0],
                "premium": row[1],
                "quota": row[2],
                "bids_received": row[3],
                "bids_success": row[4],
                "month": str(row[5])[:7],
            }
            for row in result
        ],
    }


@router.get("/category/{category_name}")
def get_category_trend(category_name: str, db: Session = Depends(get_db)):
    """Get full trend for a specific COE category."""
    valid = ["Category A", "Category B", "Category C", "Category D", "Category E"]
    if category_name not in valid:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid category. Choose from: {valid}"
        )

    result = db.execute(text("""
        SELECT month, premium, quota, bids_received, bids_success
        FROM coe_results
        WHERE vehicle_class = :cat
        ORDER BY month
    """), {"cat": category_name}).fetchall()

    premiums = [r[1] for r in result]

    return {
        "category": category_name,
        "count": len(result),
        "stats": {
            "min": min(premiums),
            "max": max(premiums),
            "avg": round(sum(premiums) / len(premiums), 2),
            "latest": premiums[-1],
        },
        "data": [
            {
                "month": str(r[0])[:7],
                "premium": r[1],
                "quota": r[2],
                "bids_received": r[3],
                "bids_success": r[4],
            }
            for r in result
        ],
    }


@router.get("/summary/stats")
def get_summary_stats(db: Session = Depends(get_db)):
    """Get aggregate statistics across all categories."""
    result = db.execute(text("""
        SELECT
            vehicle_class,
            COUNT(*) as total_records,
            MIN(premium) as min_premium,
            MAX(premium) as max_premium,
            AVG(premium) as avg_premium,
            MIN(month) as earliest,
            MAX(month) as latest
        FROM coe_results
        GROUP BY vehicle_class
        ORDER BY vehicle_class
    """)).fetchall()

    return {
        "data": [
            {
                "category": row[0],
                "total_records": row[1],
                "min_premium": round(row[2], 0),
                "max_premium": round(row[3], 0),
                "avg_premium": round(row[4], 0),
                "date_range": f"{str(row[5])[:7]} → {str(row[6])[:7]}",
            }
            for row in result
        ]
    }