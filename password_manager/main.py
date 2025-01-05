import os
import sqlite3
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
import secrets
import base64

# Constants
DATABASE = "password_manager.db"

# Key Derivation
def derive_key(password, salt):
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100_000,
        backend=default_backend()
    )
    return kdf.derive(password.encode())

# Encrypt function
def encrypt(data, key):
    iv = os.urandom(16)  # Initialization vector
    cipher = Cipher(algorithms.AES(key), modes.CFB(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(data.encode()) + encryptor.finalize()
    return base64.b64encode(iv + ciphertext).decode()

# Decrypt function
def decrypt(data, key):
    raw = base64.b64decode(data.encode())
    iv, ciphertext = raw[:16], raw[16:]
    cipher = Cipher(algorithms.AES(key), modes.CFB(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    return (decryptor.update(ciphertext) + decryptor.finalize()).decode()

# Generate random password
def generate_password(length=16):
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()"
    return ''.join(secrets.choice(alphabet) for _ in range(length))

# Database setup
def setup_database():
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS passwords (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            service TEXT NOT NULL,
            username TEXT NOT NULL,
            password TEXT NOT NULL
        )
        """)
        conn.commit()

# Add password
def add_password(service, username, password, key):
    encrypted_password = encrypt(password, key)
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute("INSERT INTO passwords (service, username, password) VALUES (?, ?, ?)",
                       (service, username, encrypted_password))
        conn.commit()

# Show passwords
def show_passwords(key):
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT id, service, username, password FROM passwords")
        rows = cursor.fetchall()
        for row in rows:
            decrypted_password = decrypt(row[3], key)
            print(f"{row[0]}: Service: {row[1]}, Username: {row[2]}, Password: {decrypted_password}")

# Delete password
def delete_password(entry_id):
    with sqlite3.connect(DATABASE) as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM passwords WHERE id = ?", (entry_id,))
        conn.commit()

# Main Menu
def main():
    # Key generation
    master_password = input("Set your master password: ")
    salt = os.urandom(16)
    key = derive_key(master_password, salt)
    
    setup_database()

    while True:
        print("\nPassword Manager")
        print("1. Add Password")
        print("2. Show Passwords")
        print("3. Delete Password")
        print("4. Generate Password")
        print("5. Exit")
        choice = input("Choose an option: ")

        if choice == "1":
            service = input("Enter service name: ")
            username = input("Enter username: ")
            password = input("Enter password (or leave blank to generate): ")
            if not password:
                password = generate_password()
                print(f"Generated password: {password}")
            add_password(service, username, password, key)
        elif choice == "2":
            show_passwords(key)
        elif choice == "3":
            entry_id = input("Enter the ID to delete: ")
            delete_password(entry_id)
        elif choice == "4":
            length = int(input("Enter password length: "))
            print(f"Generated password: {generate_password(length)}")
        elif choice == "5":
            break
        else:
            print("Invalid choice. Try again.")

if __name__ == "__main__":
    main()
