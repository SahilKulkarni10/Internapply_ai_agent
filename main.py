import asyncio
import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import SecretStr

from browser_use import Agent, BrowserConfig
from browser_use.browser.browser import Browser
from browser_use.browser.context import BrowserContextConfig


load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')
if not api_key:
    raise ValueError('GEMINI_API_KEY is not set')


internshala_username = os.getenv('INTERNSHALA_USERNAME')
internshala_password = os.getenv('INTERNSHALA_PASSWORD')


llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp', api_key=SecretStr(api_key))


browser = Browser(
    config=BrowserConfig(
        new_context_config=BrowserContextConfig()
    )
)


resume_path = "/Users/sahilkulkarni/Desktop/resume_agent/RenderCV_sb2nov_Theme-3.pdf"
if not os.path.exists(resume_path):
    raise FileNotFoundError(f"Resume file not found at {resume_path}")

async def apply_for_internships():
    
    if internshala_username and internshala_password:
        
        task = (
            "Go to internshala.com, log in with username {internshala_username} and password {internshala_password}, "
            "search for 'web development internship 2025', filter by 'Work from Home' if possible, "
            "find an internship posting specifically for web development, click 'Apply Now', "
            "answer basic questions (e.g., 'Why should we hire you?' with 'I have strong web development skills "
            "in HTML, CSS, and JavaScript, and I’m eager to contribute and learn,' or 'Do you have relevant skills?' "
            "with 'Yes'), upload the resume located at {resume_path}, and submit the application. "
            "Ignore any advertisements, pop-ups, or offers (e.g., training or subscription prompts). "
            "If the website https://isp.internshala.com/ opens or any posting related to 'Internshala Student Partner' "
            "appears, close that page immediately, do not reopen it, and return to internshala.com to continue. "
            "Skip any non-web development roles. "
            "If a CAPTCHA appears with an error message like 'CAPTCHA error, please try again', "
            "click the close button (e.g., 'Close' or 'X') and proceed to the next step or internship posting. "
            "Skip any posting requiring complex steps beyond basic questions and resume upload (e.g., cover letter). "
            "Apply to exactly one web development internship on internshala.com and then stop."
        ).format(internshala_username=internshala_username, internshala_password=internshala_password, resume_path=resume_path)
    else:
        
        task = (
            "Go to internshala.com, search for 'web development internship 2025', "
            "filter by 'Work from Home' if possible, and list the title, company, and URL "
            "of the first web development internship posting you find. "
            "Ignore any advertisements, pop-ups, or offers (e.g., training or subscription prompts). "
            "If the website https://isp.internshala.com/ opens or any posting related to 'Internshala Student Partner' "
            "appears, close that page immediately, do not reopen it, and return to internshala.com to continue. "
            "Skip any non-web development roles. "
            "If a CAPTCHA appears with an error message like 'CAPTCHA error, please try again', "
            "click the close button (e.g., 'Close' or 'X') and continue without stopping. "
            "Do not attempt to apply or log in."
        )

  
    agent = Agent(
        task=task,
        llm=llm,
        max_actions_per_step=4,
        browser=browser,
    )

   
    await agent.run(max_steps=50)  

if __name__ == '__main__':
    asyncio.run(apply_for_internships())