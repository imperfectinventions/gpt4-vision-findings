import sys
import base64
import requests
from PIL import Image, ImageDraw, ImageFont
import openai 
import json
import io
import settings

class coordinate_handler():
    default_settings_dict = {
        "line_params" : {
            "fill_color" : (0, 0, 0), 
            "thickness"  : 10,
            "start_pos"  : (0, 0)
        },
        "font_params" : {
            "font_file" : "arial.ttf",
            "font_size" : 20,
            "fill_color": (0, 0, 0),
        },
        "im_params" : {
            "blank_im_color" : (255, 255, 255)
        }
    }
    def build_settings(self, build_settings_dict):
        compare_dict = coordinate_handler.default_settings_dict
        if len(self.settings_dict) > 0:
            compare_dict = self.settings_dict
        #set the default settings when values are not passed in
        if len(build_settings_dict) > 0:
            for def_param in compare_dict:
                if def_param in build_settings_dict:
                    self.settings_dict[def_param] = {}
                    for nest_param in compare_dict:
                        if nest_param in build_settings_dict[def_param]:
                            self.settings_dict[def_param][nest_param] = build_settings_dict[def_param][nest_param]
                        else:
                            self.settings_dict[def_param][nest_param] = compare_dict[def_param][nest_param]
                else:
                    self.settings_dict[def_param] = compare_dict[def_param]
        else:
            self.settings_dict = compare_dict

    def __init__(self, settings_dict={}):
        self.settings_dict = {}
        self.build_settings(settings_dict)

    def gen_coords(self, image_in, num_rows, num_cols, settings_override={}):
        '''Overlays the desired coordinates on top of the image.'''
        blank_img = Image.new('RGBA', image_in.size, self.settings_dict["im_params"]["blank_im_color"])
        image_in = image_in.resize((image_in.size[0] - self.settings_dict["font_params"]["font_size"], image_in.size[1] - self.settings_dict["font_params"]["font_size"]))
        blank_img.paste(image_in, (self.settings_dict["font_params"]["font_size"], self.settings_dict["font_params"]["font_size"]))
        image_in = blank_img
        font = ImageFont.truetype(self.settings_dict["font_params"]["font_file"], size=self.settings_dict["font_params"]["font_size"])
        if len(settings_override) > 0:
            self.build_settings(settings_override)
        draw_im = ImageDraw.Draw(image_in)
        assert(num_rows < 12 and num_cols < 12) #setting this limit as right now, more coordinates isn't really better from testing
        for row in range(0, num_rows):
            slice = image_in.size[0]//num_rows
            curr_cut = image_in.size[0]*row//num_rows - self.settings_dict["line_params"]["thickness"] if (image_in.size[0]*row//num_rows - self.settings_dict["line_params"]["thickness"]) > 0 else 0
            #lines are always num_x - 1
            if row > 0:
                draw_im.line((curr_cut, self.settings_dict["line_params"]["start_pos"][1], curr_cut, image_in.size[1]), fill=self.settings_dict["line_params"]["fill_color"][1], width=self.settings_dict["line_params"]["thickness"])
            #setting the font
            draw_im.text((curr_cut+slice//3, self.settings_dict["line_params"]["start_pos"][1]), str(row+1), font=font, fill=self.settings_dict["font_params"]["fill_color"])
        for col in range(0, num_cols):
            slice = image_in.size[1]//num_cols
            curr_cut = image_in.size[1]*col//num_cols - self.settings_dict["line_params"]["thickness"] if (image_in.size[1]*col//num_cols - self.settings_dict["line_params"]["thickness"]) > 0 else 0
            if col > 0:
                draw_im.line((self.settings_dict["line_params"]["start_pos"][0], curr_cut, image_in.size[0], curr_cut), fill=self.settings_dict["line_params"]["fill_color"], width=self.settings_dict["line_params"]["thickness"])
            draw_im.text((self.settings_dict["line_params"]["start_pos"][0], curr_cut+slice//3), chr(65+col), font=font, fill=self.settings_dict["font_params"]["fill_color"])
    
        return image_in 
    
    def gen_base64_im(self, image_in):
        '''Generate the base64 image that will go into GPT-4 Vision'''
        image_bytes = io.BytesIO()
        (image_in.convert("RGB")).save(image_bytes, format="JPEG")
        image_bytes = image_bytes.getvalue()
        upload_data = base64.b64encode(image_bytes).decode('utf-8')
        return upload_data

#need to set the openai.api_key first before use
def check_have_gpt_vision():
    '''check to see if you have gpt vision'''
    for i in openai.models.list().data:
        if "gpt-4-vision-preview" in i.id:
            return True
    return False

def coord_gpt4(image_file_names, prompt, sys_msg="", model="gpt-4-vision-preview", max_tokens=300, temperature=0, show=True, x_nums = (2, 2), settings_override={}):
    coord_handle = coordinate_handler(settings_override)
    openai.api_key = settings.api_key
    if not check_have_gpt_vision():
        raise Exception("You don't have GPT-4. You need to buy some credits and generate a new API key")
    content = [{}]
    ims = []
    for image_file_name in image_file_names:
        im = Image.open(image_file_name)
        coord_im = coord_handle.gen_coords(im, x_nums[0], x_nums[1])
        ims.append(coord_im)
        content.append({
            "type": "image_url",
            "image_url": {
              "url": f"data:image/jpeg;base64,{coord_handle.gen_base64_im(coord_im)}"
            }
          })
    messages = []
    if len(sys_msg) > 0:
        messages.append({
            "role" : "system",
            "content" : sys_msg
        })
    content[0] = {
        "type" : "text",
        "text" : prompt,
    }
    messages.append({
        "role" : "user",
        "content" : content
    })
    client = openai.OpenAI(api_key = settings.api_key)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature
    )
    if show:
        for i in ims:
            i.show()
    return response.choices[0]

print(
        coord_gpt4([r"C:\Users\jerem\Downloads\fiish.jpg"], 
           "What coordinate is the jumping fish in? Explain your reasoning.", 
           sys_msg="You analyze coordinates of images and try to identify the coordinates that the requested item is in. If an object is in multiple coordinates, then identify the multiple coordinates when possible. You always explain your reasoning.", 
           model="gpt-4-vision-preview").message.content
)