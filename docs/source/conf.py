# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.abspath("../.."))
# 添加src目录到路径
sys.path.insert(0, os.path.abspath("../../src"))


project = 'MetaBox'
copyright = '2025, MetaEvo'
author = 'MetaEvo'
release = 'v2.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.linkcode',
    'autodoc2',
    # 'sphinx.ext.viewcode',
    'sphinx_markdown_tables',
    # 'myst_parser',
    "myst_nb",
    'sphinx_copybutton',
    
]
viewcode_follow_imported_members = True
viewcode_enable_epub = True
autodoc2_config = {
    "add_module_names": True,
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "inherited-members": "base",
    "recursive": True,
    "special-members": "__init__",
    # "linkcode_resolve": True,  # 启用链接到源码
    # "linkcode": True,  # 启用链接到源码
}
def linkcode_resolve(domain, info):
    if domain != 'py':
        return None
    if not info['module']:
        return None
    
    # 尝试查找源文件
    module = info['module']
    fullname = info['fullname']
    print(f"Resolving link for module: {module}, fullname: {fullname}")
    # 构建GitHub链接或本地文件路径
    # 这里替换为你的GitHub仓库URL或自定义路径生成逻辑
    return f"https://github.com/GMC-DRL/MetaBox/blob/Final/{module.replace('.', '/')}.py"
autodoc2_packages = [
    {
        "path": "../../src",
        # "auto_mode": False,
    },
]
# 设置autodoc2输出Markdown格式
autodoc2_output_format = "myst"

# 确保输出目录存在并已配置
autodoc2_output_dir = "apidocs"

autodoc2_docstring_parser_regexes = [
    # this will render all docstrings as Markdown
    (r".*", "myst"),
    # # this will render select docstrings as Markdown
    # (r"autodoc2\..*", "myst"),
]
autodoc2_render_plugin = "myst"
# autodoc2_module_all_regexes = [
#     r"src",
    
# ]

templates_path = ['_templates']
exclude_patterns = []

# 添加 MyST 解析器配置
myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "fieldlist",
    "linkify",
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "shibuya"
html_context = {
    "source_type": "github",
    "source_user": "MetaEvo",
    "source_repo": "MetaBox",
    "source_version": "v2.0.0",
    "source_docs_path": "/docs/source/", 
}
html_theme_options = {
    "github_url": "https://github.com/MetaEvo/MetaBox",
}
# html_theme = 'sphinx_book_theme'
# html_theme = 'press'
html_static_path = ['_static']
locale_dirs = ["locale/"]
gettext_compact = "docs"
autosummary_generate = True
autosummary_imported_members = True
nb_execution_mode = "off"
nbsphinx_execute = "never"
nb_output_stderr = "remove"
