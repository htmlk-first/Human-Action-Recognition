import React from "react";
import { recommend } from "../services/api";
import { exerciseList } from "../services/exerciseList";

const exerciseCard = (name, imgSrc, num) => {
	return (
		<div className="col" key={num}>
			<div className="card">
				<img
					src={`images/${imgSrc}`}
					className="card-img-top"
					alt={name}
				/>
				<div className="card-body">
					<h5 className="card-title">{name}</h5>
					<p className="card-text"></p>
					<a
						href="https://kw-kwix.github.io/pose-detector/"
						rel="noreferrer"
						target={"_blank"}
					>go</a>
				</div>
			</div>
		</div>
	);
};

export function Recommend() {
	const [cardList, setCardList] = React.useState(null);
	React.useEffect(() => {
		const email = localStorage.getItem("kwixUser");
		recommend(email)
			.then((res) => {
				for (const key in res.data.result) {
					if (Object.hasOwnProperty.call(res.data.result, key)) {
						exerciseList[key]["score"] = res.data.result[key];
					}
				}
				const result = exerciseList.sort((a, b) => b.score - a.score);
				console.log(result);
				setCardList(
					[...Array(6).keys()].map((i) =>
						exerciseCard(result[i].name, result[i].imgSrc, i)
					)
				);
			})
			.catch((res) => alert(res));
	}, []);
	return (
		<div>
			<div className="container">
				<h1>추천 운동 목록</h1>
			</div>
			<div className="container">
				<div className="row row-cols-1 row-cols-md-3 g-4">
					{cardList}
				</div>
			</div>
		</div>
	);
}
