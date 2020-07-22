import twint

c= twint.Config()
c.Search = "#earthquake"
c.Since = "2018-12-01 20:30:15"
c.Until = "2019-12-01 20:39:15"
c.Near = "Rome"
c.Output = "tweets.csv"
c.Store_csv = True
c.Format = "ID {id} | Username {username} | Date {date}"

twint.run.Search(c)