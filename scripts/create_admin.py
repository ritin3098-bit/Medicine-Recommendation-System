"""Create the first administrator (or promote an existing account).

  python -m scripts.create_admin                      # interactive (password is not echoed)
  python -m scripts.create_admin --promote alice      # make an existing user an admin

Alternatively set MEDREC_ADMIN_USER / MEDREC_ADMIN_EMAIL / MEDREC_ADMIN_PASSWORD before starting the app; the admin is
created on first start if none exists.
"""
import argparse
import getpass
import sys

from src import auth
from src.db import connect


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--promote", metavar="USERNAME", help="promote an existing account to admin")
    ap.add_argument("--username")
    ap.add_argument("--email")
    args = ap.parse_args()

    if args.promote:
        with connect() as c:
            n = c.execute("UPDATE users SET role='admin' WHERE username=?", (args.promote.strip().lower(),)).rowcount
        sys.exit(f"{args.promote} is now an admin." if n else f"No user named '{args.promote}'.")

    username = args.username or input("Admin username: ")
    email = args.email or input("Admin email: ")
    pw = getpass.getpass("Password: ")
    if pw != getpass.getpass("Confirm password: "):
        sys.exit("Passwords do not match.")
    try:
        user, code = auth._create_user(username, email, "Administrator", pw, pw, True, "admin")
    except auth.ValidationError as e:
        sys.exit("Could not create admin: " + " ".join(e.errors))
    print(f"Admin '{user['username']}' created.\nRecovery code (store it safely, shown once): {code}")


if __name__ == "__main__":
    main()
