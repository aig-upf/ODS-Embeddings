#!/bin/bash

TEMP_PATH="TemporaryYnsaneDatasetDownload"
URL="$1"
FORMAT="$2"
DESTINATION="$3"
DEST_CMD="$4"

# clean up the paths
rm -f "$TEMP_PATH"
rm -rf "$DESTINATION"

# download files
wget --quiet "$URL" -O "$TEMP_PATH"
echo "Downloaded '$URL'."

# decompress and move files
if [ "$FORMAT" = "zip" ]; then
  echo "Extracting files to '$DESTINATION' -- Zip compressed."
  unzip "$TEMP_PATH" -d "$DESTINATION"
  rm "$TEMP_PATH"
elif [ "$FORMAT" = "gz" ]; then
  echo "Extracting files to '$DESTINATION' -- Gzip compressed."
  zcat < "$TEMP_PATH" > "$DESTINATION"
  rm "$TEMP_PATH"
else
  echo "Format is neither zip nor gz, moving to '$DESTINATION'."
  mv "$TEMP_PATH" "$DESTINATION"
fi

# run additional commands, if needed
if ! [ -z "$DEST_CMD" ]; then
  echo "Running additional commands in '$3'"
  cd "$DESTINATION"
  bash -c "$DEST_CMD"
fi

