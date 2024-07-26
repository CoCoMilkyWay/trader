# Add the repository and update the package list
# sudo add-apt-repository ppa:ubuntu-toolchain-r/test
# sudo apt update

# Install essential build tools and libraries
sudo apt-get install -y git cmake build-essential libgmp-dev libmpfr-dev libmpc-dev zlib1g-dev

# Function to download a package if it does not exist
download_if_not_exists() {
  local url=$1
  local file=$(basename $url)
  if [ ! -f $file ]; then
    wget $url
  else
    echo "$file already exists, skipping download."
  fi
}

# Download the pre-compiled GCC and G++ packages and their dependencies
sudo apt update
download_if_not_exists http://mirrors.kernel.org/ubuntu/pool/universe/g/gcc-8/gcc-8_8.4.0-3ubuntu2_amd64.deb
download_if_not_exists http://mirrors.edge.kernel.org/ubuntu/pool/universe/g/gcc-8/gcc-8-base_8.4.0-3ubuntu2_amd64.deb
download_if_not_exists http://mirrors.kernel.org/ubuntu/pool/universe/g/gcc-8/libgcc-8-dev_8.4.0-3ubuntu2_amd64.deb
download_if_not_exists http://mirrors.kernel.org/ubuntu/pool/universe/g/gcc-8/cpp-8_8.4.0-3ubuntu2_amd64.deb
download_if_not_exists http://mirrors.kernel.org/ubuntu/pool/universe/g/gcc-8/libmpx2_8.4.0-3ubuntu2_amd64.deb
download_if_not_exists http://mirrors.kernel.org/ubuntu/pool/main/i/isl/libisl22_0.22.1-1_amd64.deb
sudo apt install ./libisl22_0.22.1-1_amd64.deb ./libmpx2_8.4.0-3ubuntu2_amd64.deb ./cpp-8_8.4.0-3ubuntu2_amd64.deb ./libgcc-8-dev_8.4.0-3ubuntu2_amd64.deb ./gcc-8-base_8.4.0-3ubuntu2_amd64.deb ./gcc-8_8.4.0-3ubuntu2_amd64.deb

download_if_not_exists http://mirrors.kernel.org/ubuntu/pool/universe/g/gcc-8/libstdc++-8-dev_8.4.0-3ubuntu2_amd64.deb
download_if_not_exists http://mirrors.kernel.org/ubuntu/pool/universe/g/gcc-8/g++-8_8.4.0-3ubuntu2_amd64.deb
sudo apt install ./libstdc++-8-dev_8.4.0-3ubuntu2_amd64.deb ./g++-8_8.4.0-3ubuntu2_amd64.deb

# Fix broken dependencies, if any
sudo apt-get install -f

# Verify the installation
gcc-8 --version
g++-8 --version

# Clean up downloaded .deb files (optional)
rm *.deb.*

# Set the alternatives to switch between GCC versions
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 800 --slave /usr/bin/g++ g++ /usr/bin/g++-8
sudo update-alternatives --config gcc
sudo update-alternatives --config g++

# Optionally, remove old symbolic links and create new ones for gcc and g++
# sudo rm /usr/bin/gcc
# sudo ln -s /usr/bin/gcc-8 /usr/bin/gcc
# sudo ln -s /usr/bin/g++-8 /usr/bin/g++

ll -h /usr/bin/gcc*
ll -h /usr/bin/g++*