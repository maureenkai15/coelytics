from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, text
from sqlalchemy.orm import declarative_base, sessionmaker
from pathlib import Path
import pandas as pd
from datetime import datetime

DB_PATH = Path("coelytics.db")
DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class COERecord(Base):
    __tablename__ = "coe_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    month = Column(DateTime, nullable=False, index=True)
    vehicle_class = Column(String(20), nullable=False, index=True)
    premium = Column(Float, nullable=False)
    quota = Column(Integer, nullable=True)
    bids_received = Column(Integer, nullable=True)
    bids_success = Column(Integer, nullable=True)
    bidding_no = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<COERecord {self.month.strftime('%Y-%m')} {self.vehicle_class} ${self.premium:,.0f}>"


def init_db():
    Base.metadata.create_all(bind=engine)
    print("Database tables created.")


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def safe_int(val):
    """Convert a value to int, handling commas and NaN."""
    if pd.isna(val):
        return None
    return int(str(val).replace(",", "").strip())


def load_csv_to_db(csv_path: str = "data/processed/coe_clean.csv"):
    print(f"Loading {csv_path} into database...")

    df = pd.read_csv(csv_path, parse_dates=["month"])

    if "_id" in df.columns:
        df = df.drop(columns=["_id"])

    db = SessionLocal()
    try:
        db.execute(text("DELETE FROM coe_results"))
        db.commit()

        records = []
        for _, row in df.iterrows():
            record = COERecord(
                month=row["month"],
                vehicle_class=row["vehicle_class"],
                premium=float(str(row["premium"]).replace(",", "")),
                quota=safe_int(row.get("quota")),
                bids_received=safe_int(row.get("bids_received")),
                bids_success=safe_int(row.get("bids_success")),
                bidding_no=safe_int(row.get("bidding_no")),
            )
            records.append(record)

        db.bulk_save_objects(records)
        db.commit()
        print(f"Inserted {len(records)} records into coe_results.")

    finally:
        db.close()


def query_summary():
    db = SessionLocal()
    try:
        total = db.execute(text("SELECT COUNT(*) FROM coe_results")).scalar()
        latest = db.execute(text("SELECT MAX(month) FROM coe_results")).scalar()
        earliest = db.execute(text("SELECT MIN(month) FROM coe_results")).scalar()
        categories = db.execute(
            text("SELECT DISTINCT vehicle_class FROM coe_results ORDER BY vehicle_class")
        ).fetchall()

        print(f"\nDatabase summary:")
        print(f"  Total records : {total}")
        print(f"  Date range    : {earliest[:7]} → {latest[:7]}")
        print(f"  Categories    : {[c[0] for c in categories]}")

        print("\nLatest premium by category:")
        result = db.execute(text("""
            SELECT vehicle_class, premium, month
            FROM coe_results
            WHERE month = (SELECT MAX(month) FROM coe_results)
            ORDER BY vehicle_class
        """)).fetchall()
        for row in result:
            print(f"  {row[0]:<12} ${row[1]:>10,.0f}  ({str(row[2])[:7]})")
    finally:
        db.close()


if __name__ == "__main__":
    init_db()
    load_csv_to_db()
    query_summary()