import json
import os
import argparse

VERSION_FILE = os.path.join('models', 'version_info.json')

def get_current_version():
    """Get the current version from file or create default if not exists."""
    if not os.path.exists(VERSION_FILE):
        version = {"major": 1, "minor": 0, "patch": 0}
        os.makedirs(os.path.dirname(VERSION_FILE), exist_ok=True)
        with open(VERSION_FILE, 'w') as f:
            json.dump(version, f, indent=4)
        return version
    
    with open(VERSION_FILE, 'r') as f:
        return json.load(f)

def save_version(version):
    """Save the version to the version file."""
    os.makedirs(os.path.dirname(VERSION_FILE), exist_ok=True)
    with open(VERSION_FILE, 'w') as f:
        json.dump(version, f, indent=4)

def increment_version(part):
    """Increment the specified version part."""
    version = get_current_version()
    
    if part == "major":
        version["major"] += 1
        version["minor"] = 0
        version["patch"] = 0
    elif part == "minor":
        version["minor"] += 1
        version["patch"] = 0
    elif part == "patch":
        version["patch"] += 1
    
    save_version(version)
    return version

def set_version(major=None, minor=None, patch=None):
    """Set version to specific values."""
    version = get_current_version()
    
    if major is not None:
        version["major"] = major
    if minor is not None:
        version["minor"] = minor
    if patch is not None:
        version["patch"] = patch
    
    save_version(version)
    return version

def print_version(version):
    """Print version in a readable format."""
    print(f"Current version: MPP_v{version['major']}.{version['minor']}.{version['patch']}")

def main():
    parser = argparse.ArgumentParser(description="Manage model versioning")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Show command
    show_parser = subparsers.add_parser("show", help="Show current version")
    
    # Increment command
    increment_parser = subparsers.add_parser("increment", help="Increment version")
    increment_parser.add_argument("part", choices=["major", "minor", "patch"], 
                                  help="Which part of the version to increment")
    
    # Set command
    set_parser = subparsers.add_parser("set", help="Set version to specific values")
    set_parser.add_argument("--major", type=int, help="Set major version")
    set_parser.add_argument("--minor", type=int, help="Set minor version")
    set_parser.add_argument("--patch", type=int, help="Set patch version")
    
    args = parser.parse_args()
    
    if args.command == "show" or args.command is None:
        version = get_current_version()
        print_version(version)
    
    elif args.command == "increment":
        version = increment_version(args.part)
        print(f"Version incremented to: MPP_v{version['major']}.{version['minor']}.{version['patch']}")
    
    elif args.command == "set":
        if not any([args.major, args.minor, args.patch]):
            print("Error: At least one version component (major, minor, patch) must be specified")
            return
        
        version = set_version(args.major, args.minor, args.patch)
        print(f"Version set to: MPP_v{version['major']}.{version['minor']}.{version['patch']}")

if __name__ == "__main__":
    main() 