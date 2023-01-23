"""
Ratei displays two images. 
The user presses either left or right to choose which image is better.
Ratei then updates the ELO rating of each image.
The images are then reshuffled and displayed again.

Images are resaved with a new name based on the ELO rating.
"""


import os
import shutil
import sys
import random
import math
import pygame
from pygame.locals import *

from use_openai_text import OpenAIConvo

from colorize import colorize

image_dir = 'Tiki Lounge Mood Board'
# get the images
images = os.listdir(image_dir)
# get fraction of images to use
fraction = 1.0
# fraction = 0.5
# fraction = 0.25
# fraction = 0.125
images = images[:math.ceil(len(images)*fraction)]
print('Using', colorize(len(images), 'yellow'), 'images')

prompt_file = 'filename_summarizer_prompt.txt'
with open(prompt_file, 'r') as f:
    prompt = f.read()

# model = 'text-davinci-003'
model = 'text-babbage-001'
filename_summaries = {
    image: OpenAIConvo.single_prompt(
        prompt=prompt+str(image)+'\nFolder:',
        model=model,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        temperature=0.05,
        max_tokens=100,
        verbose=False,
    )[0].strip()+'_'+str(i) for i, image in enumerate(images)
}

for image, summary in filename_summaries.items():
    # print(colorize(image, 'yellow') + ': ' + colorize(summary, 'green'))
    print(colorize(summary, 'green'))

# exit()

# shuffle the images
random.shuffle(images)

# replace file names with id numbers
image_ids = {image: filename_summaries[image] for image in images}

# image ratings (ELO rating system)
# https://en.wikipedia.org/wiki/Elo_rating_system
# jitter = int(math.sqrt(len(images)))
jitter = 0
ratings = {image: random.randint(-jitter, jitter)+1000 for image in images}

# initialize pygame
pygame.init()
# set the window size
window_size = (800, 400)
screen = pygame.display.set_mode(window_size)
# set the window title
pygame.display.set_caption('Ratei')
# set the background color
background_color = (255, 255, 255)
# font
font = pygame.font.SysFont('Arial', 20)
# colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (127, 127, 127)
LIGHT_GREY = (191, 191, 191)


def write_clear(text, x, y):
    assert type(text) == str, 'text must be a string not a ' + str(type(text))
    # convert text to unicode
    text = str(text).strip()
    # write text
    for x_offset in range(-1, 2):
        for y_offset in range(-1, 2):
            if x_offset == 0 and y_offset == 0:
                continue
            to_draw = font.render(text, True, BLACK)
            screen.blit(to_draw, (x+x_offset, y+y_offset))
    to_draw = font.render(text, True, LIGHT_GREY)
    screen.blit(to_draw, (x, y))


def draw_images(left, right):
    # draw the left image
    left_image = pygame.image.load(image_dir + '/' + left)
    left_image = pygame.transform.scale(left_image, (400, 400))
    screen.blit(left_image, (0, 0))
    # draw image summary
    write_clear(filename_summaries[left], 10, 10)
    # draw rating
    write_clear(f'{ratings[left]:.0f}', 10, 30)
    

    # draw the right image
    right_image = pygame.image.load(image_dir + '/' + right)
    right_image = pygame.transform.scale(right_image, (400, 400))
    screen.blit(right_image, (400, 0))
    # draw image summary
    write_clear(filename_summaries[right], 410, 10)
    # draw rating
    write_clear(f'{ratings[right]:.0f}', 410, 30)


def get_matchup():
    for image in images:
        for image2 in images:
            if image != image2:
                yield image, image2

def set_beats(winner, loser):
    # update ratings
    winner_rating = ratings[winner]
    loser_rating = ratings[loser]
    expected_winner = 1 / (1 + 10 ** ((loser_rating - winner_rating) / 400))
    expected_loser = 1 / (1 + 10 ** ((winner_rating - loser_rating) / 400))
    ratings[winner] = winner_rating + 32 * (1 - expected_winner)
    ratings[loser] = loser_rating + 32 * (0 - expected_loser)


def do_files():
    out_dir = 'Ratei Rankings'
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    # delete old rankings
    for file in os.listdir(out_dir):
        os.remove(os.path.join(out_dir, file))
    # write new rankings (ranking + _ + id)
    for image in images:
        new_filename = str(round(ratings[image])).zfill(4) + '_' + image_ids[image] + '.png'
        # copy image to new directory with new filename
        old_image_path = os.path.join(image_dir, str(image))
        new_image_path = os.path.join(out_dir, new_filename)
        shutil.copy(old_image_path, new_image_path)
    

matchups = list(get_matchup())
all_matchups = matchups.copy()
random.shuffle(matchups)

seen_matchups = set()
seen_temp = {image: 0 for image in images}

def cull_matchups():
    # remove matchups that have already been done
    for matchup in all_matchups:
        if matchup in seen_matchups and matchup in matchups:
            matchups.remove(matchup)
        if matchup[::-1] in seen_matchups and matchup in matchups:
            matchups.remove(matchup)
    # remove matchups where the rating difference is too great
    for matchup in matchups:
        if abs(ratings[matchup[0]] - ratings[matchup[1]]) > 120:
            matchups.remove(matchup)
    def rating_diff(matchup):
        # prefer both images to have a rating of 1000
        # if ratings[matchup[0]] == 1000 and ratings[matchup[1]] == 1000:
            # return float('-inf')
        matchup_similarity = abs(ratings[matchup[0]] - ratings[matchup[1]]) + 1
        matchup_max = max(ratings[matchup[0]], ratings[matchup[1]]) + 1
        matchup_min = min(ratings[matchup[0]], ratings[matchup[1]]) + 1
        metric = matchup_similarity / matchup_max**2 / matchup_min
        return metric
    matchups.sort(key=rating_diff, reverse=True)

left_image, right_image = matchups.pop()
seen_temp[left_image] += 1
seen_temp[right_image] += 1
seen_matchups.add((left_image, right_image))

draw_images(left_image, right_image)
pygame.display.update()

# main loop
while True:
    # check for quit event
    for event in pygame.event.get():
        if event.type == QUIT:
            pygame.quit()
            sys.exit()
        if event.type == KEYDOWN:
            if event.key == K_LEFT:
                set_beats(left_image, right_image)

            elif event.key == K_RIGHT:
                set_beats(right_image, left_image)
            cull_matchups()
                
            left_image, right_image = matchups.pop()
            seen_temp[left_image] += 1
            seen_temp[right_image] += 1 
            seen_matchups.add((left_image, right_image))

            for image in seen_temp:
                seen_temp[image] -= 0.1
            print({image_ids[image]: round(ratings[image]) for image in sorted(ratings, key=ratings.get, reverse=True)})
            # print(f'Comparing {left_image} and {right_image}')

            do_files()
            if not matchups:
                print('Done!')
                pygame.quit()
                sys.exit()

            # draw the images
            draw_images(left_image, right_image)
            matchups_to_go = len(matchups)
            # display number of matchups to go
            write_clear(f'{matchups_to_go//2} matchups to go', 10, 370)
            
            print(f'{matchups_to_go//2} matchups to go')

            # update the display
            pygame.display.update()
    # set the frame rate
    # delay a bit for debouncing
    pygame.time.delay(50)
    pygame.time.Clock().tick(60)
                
            


