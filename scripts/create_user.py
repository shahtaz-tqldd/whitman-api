"""Create a login user for the Whitman API.

Usage:
    python -m scripts.create_user --name "Jane Doe" --username jane --email jane@example.com
    # password is prompted when --password is omitted
"""

import argparse
import asyncio
import getpass
import sys

from sqlalchemy import or_, select

from app.core.security import hash_password
from app.db import Base
from app.db.models import User
from app.db.session import AsyncSessionLocal, engine


async def create_user(name: str, username: str, email: str, password: str) -> None:
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    async with AsyncSessionLocal() as db:
        existing = await db.scalar(
            select(User).where(or_(User.username == username, User.email == email))
        )
        if existing:
            print(f"User already exists: {existing.username} <{existing.email}>")
            sys.exit(1)

        db.add(
            User(
                name=name,
                username=username,
                email=email,
                password_hash=hash_password(password),
            )
        )
        await db.commit()

    print(f"Created user: {name} ({username} / {email})")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a Whitman API user")
    parser.add_argument("--name", required=True, help="Full name")
    parser.add_argument("--username", required=True, help="Login username")
    parser.add_argument("--email", required=True, help="Login email")
    parser.add_argument("--password", help="Password (prompted when omitted)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    password = args.password or getpass.getpass("Password: ")
    if len(password) < 8:
        print("Password must be at least 8 characters")
        sys.exit(1)
    asyncio.run(
        create_user(
            name=args.name.strip(),
            username=args.username.strip().lower(),
            email=args.email.strip().lower(),
            password=password,
        )
    )


if __name__ == "__main__":
    main()
