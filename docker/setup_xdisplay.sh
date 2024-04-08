# Run in the host machine

# eloquent_greider comes from the docker name (use `docker ps` to check)
docker cp $HOME/.Xauthority eloquent_greider:/home/purse/.Xauthority
echo $DISPLAY

# Run inside container
export DISPLAY=localhost:11.0 # should
