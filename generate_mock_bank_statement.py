import csv
import random
from datetime import datetime, timedelta

# Configuration
NUM_RECORDS = 10000
OUTPUT_FILE = "mock_bank_statement_10000.csv"

# Seed for reproducibility
random.seed(42)

# Sample descriptions and categories
income_descriptions = [
    "Salary Deposit",
    "Bonus Payment",
    "Freelance Income",
    "Investment Income",
    "Gift Received"
]

expense_descriptions = [
    "Grocery Store",
    "Restaurant",
    "Utility Bill",
    "Rent Payment",
    "Shopping",
    "Fuel Purchase",
    "Gym Membership",
    "Internet Bill",
    "Medical Expense",
    "Entertainment"
]

# Helper to generate a random date within the last year
def random_date(start_date: datetime, end_date: datetime) -> str:
    """Return a random date string between start_date and end_date (inclusive)."""
    delta = end_date - start_date
    random_days = random.randint(0, delta.days)
    date_obj = start_date + timedelta(days=random_days)
    return date_obj.strftime("%Y-%m-%d")


def generate_records(num_records: int):
    """Generate a list of mock bank statement records."""
    records = []
    today = datetime.today()
    one_year_ago = today - timedelta(days=365)

    for _ in range(num_records):
        is_income = random.random() < 0.15  # Roughly 15% of transactions are income
        if is_income:
            description = random.choice(income_descriptions)
            amount = round(random.uniform(500, 5000), 2)  # Positive income
        else:
            description = random.choice(expense_descriptions)
            amount = -round(random.uniform(5, 1000), 2)  # Negative expense

        date_str = random_date(one_year_ago, today)
        records.append({
            "date": date_str,
            "description": description,
            "amount": amount
        })
    return records


def write_csv(records, output_file):
    """Write records to a CSV file."""
    with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["date", "description", "amount"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(record)


def main():
    print(f"Generating {NUM_RECORDS} mock bank statement records...")
    records = generate_records(NUM_RECORDS)
    write_csv(records, OUTPUT_FILE)
    print(f"Mock data generated and saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main() 