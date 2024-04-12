import pickle
from pathlib import Path

import streamlit_authenticator as stauth
from streamlit_authenticator.utilities.hasher import Hasher

names = ["Abraham", "Edwin", "Rudy", "Admin" ]
usernames = ["AB", "ED","RD", "Admin"]
passwords = ["10271813", "10271851", "10280321", "Admin01"]

hashed_passwords = Hasher(passwords).generate()

file_path = Path(__file__).parent / "hashed_pw.pkl"
with file_path.open("wb") as file:
    pickle.dump(hashed_passwords, file)