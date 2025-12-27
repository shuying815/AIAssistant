import argparse
from paper import PaperManager
from image import ImageManager

def main():
    parser = argparse.ArgumentParser(description="Document and Image Management")

    subparsers = parser.add_subparsers(dest="command")

    # 论文相关命令
    paper_parser = subparsers.add_parser("add_paper", help="Add a paper to the database")
    paper_parser.add_argument("path", help="Path to the paper file")
    paper_parser.add_argument("--topics", help="Comma-separated list of topics")#required=True

    search_paper_parser = subparsers.add_parser("search_paper", help="Search for papers")
    search_paper_parser.add_argument("query", help="Search query for papers")

    # 图像相关命令
    image_parser = subparsers.add_parser("add_image", help="Add an image to the database")
    image_parser.add_argument("path", help="Path to the image")

    search_image_parser = subparsers.add_parser("search_image", help="Search for images")
    search_image_parser.add_argument("query", help="Search query for images")

    args = parser.parse_args()

    if args.command == "add_paper":
        paper_manager = PaperManager()
        if args.topics != None:
            paper_manager.add_paper(args.path, args.topics.split(","))
        else:
            paper_manager.add_paper(args.path)
        print(f"***论文 {args.path} 成功添加！")

    elif args.command == "search_paper":
        paper_manager = PaperManager()
        results = paper_manager.search_paper(args.query)
        print('***最相关的论文为:')

        # ChromaDB 返回结果结构: {'ids': [[]], 'metadatas': [[]], 'documents': [[]]}
        for i, doc in enumerate(results['documents'][0]):
            meta = results['metadatas'][0][i]
            print(f"[File] {meta['filename']} (Page {meta['page']})")
            print(f"[Snippet] {doc[:150]}...\n")

    elif args.command == "add_image":
        image_manager = ImageManager()
        image_manager.add_image(args.path)
        print(f"***图像 {args.path} 成功添加！")

    elif args.command == "search_image":
        image_manager = ImageManager()
        results = image_manager.search_image(args.query)
        print('***最相关的图像为')
        for result in results:
            print(result)

    else:
        print("***Invalid command.")

if __name__ == "__main__":
    main()
