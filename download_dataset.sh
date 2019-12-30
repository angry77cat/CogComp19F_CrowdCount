FILEID=1AHkQAeYZcQm2h1mUoYwivt1kADp238Cj
FILENAME=data.zip
wget --no-check-certificate "https://docs.google.com/uc?export=download&id=$FILEID" -O $FILENAME
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$FILEID" -O- | sed -rn "s/.*confirm=([0-9A-Za-z_]+).*/\1\n/p")&id=$FILEID" -O $FILENAME && rm -rf /tmp/cookies.txt
