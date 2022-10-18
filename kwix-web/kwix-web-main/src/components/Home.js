export function Home() {
	return (
		<div className="container">
			<div className="row">
				<div className="col-lg-12">
					<h2 className="h2-heading">Offered services</h2>
					<p className="p-heading">
						행동 인식 모델과 추천 시스템을 통해서 집에서도 할 수
						있는 홈트레이닝 보조 서비스를 제공합니다.
					</p>
				</div>
			</div>
			<div className="row">
				<div className="col-lg-4">
					<div className="text-box">
						<i className="far fa-gem"></i>
						<h4>행동 인식 모델</h4>
						<p>
							Successful online projects start with good design.
							It establishes a solid foundation for future
							development and allows for long term growth
						</p>
					</div>
				</div>
				<div className="col-lg-4">
					<div className="text-box">
						<i className="fas fa-code"></i>
						<h4>추천 시스템</h4>
						<p>
							I can code my own designs or even use the customer's
							design as base. My focus is to generate clean code
							that's well structured for reliability
						</p>
					</div>
				</div>
				<div className="col-lg-4">
					<div className="text-box">
						<i className="fas fa-tv"></i>
						<h4>홈 트레이닝</h4>
						<p>
							i can setup your project to use basic SEO principles
							which will push your project to the first page on
							search engines and save you ads money
						</p>
					</div>
				</div>
			</div>
		</div>
	);
}
